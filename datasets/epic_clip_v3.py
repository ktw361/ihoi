from typing import NamedTuple, List, Union
from argparse import ArgumentParser
import pickle, json
import os, re, bisect, time
from PIL import Image
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

from config.epic_constants import HAND_MASK_KEEP_EXPAND, EPIC_HOA_SIZE, VISOR_SIZE
from nnutils.image_utils import square_bbox
from datasets.epic_lib.epic_utils import read_v3_mask_with_occlusion

from libzhifan import io
from libzhifan.odlib import xyxy_to_xywh, xywh_to_xyxy
from libzhifan.geometry import CameraManager, BatchCameraManager

# Visualization
import matplotlib.pyplot as plt
from libzhifan import odlib
odlib.setup('xywh')
import cv2


""" Box and mask corner cases:
1. missing object mask: 
    This frame is hopeless, should be skipped.
    Set both hand_box and obj_box to None.

2. missing hoa_hand_boxes: 
    For now, skip this frame also.
    In the future: Use mask-generated hand box instead, FrankMocap might be affected?

3. missing hand_mask: 
    This is allowed, use hoa box

Object mask and box are co-processed as follows:
Generate obj_boxes (might be many), keep the one that has the largest IoU with hoa_obj_box.
If the hoa_obj_box is missing, keep the one with shorted distance.
(Future: Apply tracking to object boxes)
"""

""" Sizes:
- HOA uses 1920x1080
- visor-dense/interpolations: 854x480
- visor-dense/480p: 854x480
- visor-sparse/images: 1920x1080
- visor-sparse/masks: 854x480

previous:
- epic_analysis/interpolation: 854x480
"""

class PairLocator:
    """ locate a (vid, frame) in P01_01_0003
    See also ImageLocator and UnfilteredMaskLocator

    Interface:
    locator = PairLocator(
        result_root='/home/skynet/Zhifan/data/visor-dense/480p',
        cache_path='.cache/image_pair_index.pkl')
    path = locator.get_path(vid, frame)
    # path = <result_root>/P01_01_0003/frame_%10d.jpg
    """
    def __init__(self,
                 result_root='/home/skynet/Zhifan/data/visor-dense/480p',
                 cache_path='.cache/image_pair_index.pkl',
                 verbose=False):
        self.result_root = Path(result_root)
        self.cache_path = cache_path
        self._load_index()
        self.verbose = verbose

    def _load_index(self):
        import os.path as osp
        # cache_path = osp.join('.cache', 'pair_index.pkl')
        if not osp.exists(self.cache_path):
            os.makedirs(osp.dirname(self.cache_path), exist_ok=True)
            print("First time run, generating index...")
            _all_full_frames, _all_folders = self._build_index(
                self.result_root)
            with open(self.cache_path, 'wb') as fp:
                pickle.dump((_all_full_frames, _all_folders), fp)
            print("Index saved to", self.cache_path)

        with open(self.cache_path, 'rb') as fp:
            self._all_full_frames, self._all_folders = pickle.load(fp)

    def _build_index(self, result_root):
        import os.path as osp
        import tqdm

        def generate_pair_infos(root):
            pair_dir = os.listdir(root)
            dir_infos = []
            for d in tqdm.tqdm(pair_dir):
                l = sorted(os.listdir(osp.join(root, d)))
                mn, mx = l[0], l[-1]
                mn = int(re.search('\d{10}', mn)[0])
                mx = int(re.search('\d{10}', mx)[0])
                dir_infos.append( (d, mn, mx) )
            def func(l):
                x, _, _ = l
                a, b, c = x.split('_')
                a = a[1:]
                a = int(a)
                b = int(b)
                c = int(c)
                return a*1e7 + b*1e4 + c
            pair_infos = sorted(dir_infos, key=func)
            return pair_infos

        pair_infos = generate_pair_infos(result_root)  # pair_infos[i] = ['P01_01_0003', '123', '345']

        _all_full_frames = []
        _all_folders = []
        for folder, st, ed in pair_infos:
            min_frame = int(st)
            index = self._hash(folder, min_frame)
            _all_full_frames.append(index)
            _all_folders.append(folder)

        _all_full_frames = np.int64(_all_full_frames)
        sort_idx = np.argsort(_all_full_frames)
        _all_full_frames = _all_full_frames[sort_idx]
        _all_folders = np.asarray(_all_folders)[sort_idx]
        return _all_full_frames, _all_folders

    @staticmethod
    def _hash(vid: str, frame: int):
        pid, sub = vid.split('_')[:2]
        pid = pid[1:]
        op1, op2, op3 = map(int, (pid, sub, frame))
        index = op1 * int(1e15) + op2 * int(1e12) + op3
        return index

    def locate(self, vid, frame) -> Union[str, None]:
        """
        Returns: a str in DAVIS folder format: {vid}_{%4d}
            e.g P11_16_0107
        """
        query = self._hash(vid, frame)
        loc = bisect.bisect_right(self._all_full_frames, query)
        if loc == 0:
            return None
        r = self._all_folders[loc-1]
        r_vid = '_'.join(r.split('_')[:2])
        if vid != r_vid:
            if self.verbose:
                print(f"folder for {vid} not found")
            return None
        frames = map(
            lambda x: int(re.search('[0-9]{10}', x).group(0)),
            os.listdir(self.result_root/r))
        if max(frames) < frame:
            if self.verbose:
                print(f"Not found in {r}")
            return None
        return r

    def get_path(self, vid, frame):
        folder = self.locate(vid, frame)
        if folder is None:
            return None
        fname = f"{vid}_frame_{frame:010d}.jpg"
        fname = self.result_root/folder/fname
        return fname


class ClipInfo(NamedTuple):
    vid: str
    gt_frames: List[int]
    cat: str
    visor_name: str
    side: str  # 'left' or 'right'
    st_bound: int
    ed_bound: int
    start: int 
    end: int

    status: str  # {'UNLABELED', 'NOTFOUND_MANIP', 'NOTFOUND_OCC', 'FOUND'}
    cad: int # 0 or 1
    hand_occ: int # 0 or 1, not annotated
    obj_occ: int # 0 or 1
    bad_obj_mask: int # 0 or 1
    comments: str  # other possible comments, currently unused


class DataElement(NamedTuple):
    images: list 
    hand_bbox_dicts: list 
    side: str 
    obj_bboxes: torch.Tensor
    hand_masks: torch.Tensor 
    object_masks: torch.Tensor 
    cat: str
    global_camera: BatchCameraManager


class EpicClipDatasetV3(Dataset):

    def __init__(self,
                 image_sets,
                 all_boxes='/home/skynet/Zhifan/ihoi/weights/v3_clip_boxes.pkl',
                 cat_data_mapping='/media/skynet/DATA/Datasets/visor-dense/meta_infos/data_mapping.json',
                 image_size=VISOR_SIZE,
                 hand_expansion=0.4,
                 crop_hand_mask=True,
                 occlude_level='all',
                 sample_frames=20,
                 show_loading_time=False,
                 *args,
                 **kwargs):
        """_summary_

        Args:
            image_sets (str): path to clean set frames
                e.g. '/home/skynet/Zhifan/htmls/hos_v3_react/hos_step5_in_progress.json'
            image_size: Tuple of (W, H)
            hand_expansion (float): size of hand bounding box after squared.
            occlude_level: 'all', 'ho', 'none'. For Occlusion-aware loss
                see read_v3_mask_with_occlusion()
            crop_hand_mask: If True, will crop hand mask with only pixels
                inside hand_bbox.
            sample_frames (int):
                If clip has frames more than sample_frames,
                subsample them to a reduced number.
        """
        super().__init__(*args, **kwargs)
        self.image_size = image_size
        self.hand_expansion = hand_expansion
        self.crop_hand_mask = crop_hand_mask
        self.sample_frames = sample_frames
        self.cat_data_mapping = io.read_json(cat_data_mapping)
        self.show_loading_time = show_loading_time
        self.occlude_level = occlude_level

        # Locate frame in davis formatted folders
        self.locator = PairLocator()
        self.image_fmt = '/media/skynet/DATA/Datasets/visor-dense/480p/%s/%s_frame_%010d.jpg'  # % (folder, vid, frame)
        self.mask_fmt = '/media/skynet/DATA/Datasets/visor-dense/interpolations/%s/%s_frame_%010d.png'  # % (vid, vid, frame)

        self.box_scale = np.asarray(image_size * 2) / (EPIC_HOA_SIZE * 2)
        self.data_infos = self._read_image_sets(image_sets)
        with open(all_boxes, 'rb') as fp:
            self.ho_boxes = pickle.load(fp)

    def _read_image_sets(self, image_sets) -> List[ClipInfo]:
        """
        Some clips with wrong bounding boxes are deleted;
        Clips with comments (usually challenging ones) are deleted.

        Returns:
            list of ClipInfo(vid, nid, frame_idx, cat, side, start, end)
        """
        with open(image_sets) as fp:
            infos = json.load(fp)

        def is_valid(info: dict):
            return info.status == 'FOUND'

        infos = [ClipInfo(**v) for v in infos]
        return list(filter(is_valid, infos))

    def __len__(self):
        return len(self.data_infos)
    
    def locate_index_from_output(self, name):
        """ e.g. name = P01_09_169454_169515_action """
        arr = name.split('_')
        vid = '_'.join(arr[:2])
        start, end = int(arr[2]), int(arr[3])
        search = [i for i, v in enumerate(self.data_infos)
                  if v.vid == vid and v.start == start and v.end == end]
        return search[0]

    def _get_hand_box(self, vid, frame_idx, side, expand=True):
        hand_box = self.ho_boxes[vid][frame_idx][side]
        if not expand:
            return hand_box
        hand_box_xyxy = xywh_to_xyxy(hand_box)
        hand_box_squared_xyxy = square_bbox(
            hand_box_xyxy[None], pad=self.hand_expansion)[0]
        w, h = self.image_size
        hand_box_squared_xyxy[:2] = hand_box_squared_xyxy[:2].clip(min=[0, 0])
        hand_box_squared_xyxy[2:] = hand_box_squared_xyxy[2:].clip(max=[w, h])
        hand_box_squared = xyxy_to_xywh(hand_box_squared_xyxy)
        return hand_box_squared

    def _get_obj_box(self, vid, frame_idx, cat):
        return self.ho_boxes[vid][frame_idx][cat]

    def visualize_bboxes(self, index):
        images, hand_bbox_dicts, side, obj_bboxes, hand_masks, obj_masks, _ \
            = self.__getitem__(index)
        l = len(images)
        num_cols = 5
        num_rows = (l + num_cols - 1) // num_cols
        fig, axes = plt.subplots(
            nrows=num_rows, ncols=num_cols,
            sharex=True, sharey=True, figsize=(20, 20))
        for idx, ax in enumerate(axes.flat, start=0):
            img = images[idx]
            masked_img = img.copy()
            masked_img[hand_masks[idx] == 1, ...] = (0, 255, 0)
            masked_img[obj_masks[idx] == 1, ...] = (255, 0, 255)
            img = cv2.addWeighted(img, 0.8, masked_img, 0.2, 1.0)
            img = odlib.draw_bboxes_image_array(
                img, hand_bbox_dicts[idx][side][None], color='red')
            odlib.draw_bboxes_image(img, obj_bboxes[idx][None], color='blue')
            img = np.asarray(img)
            ax.imshow(img)
            ax.set_axis_off()
            if idx == l-1:
                break

        plt.tight_layout()
        return fig

    def _get_camera(self) -> CameraManager:
        global_cam = CameraManager(
            # fx=1050, fy=1050, cx=960, cy=540,
            fx=1050, fy=1050, cx=1280, cy=0,
            img_h=1080, img_w=1920)
        new_w, new_h = self.image_size
        global_cam = global_cam.resize(new_h=new_h, new_w=new_w)
        return global_cam
    
    def _keep_frame_with_boxes(self, vid, start, end, side, cat) -> List[int]:
        """ 
        Returns:
            a list of frames in which both obj and hand box are present
        """
        vid_boxes = self.ho_boxes[vid]
        valid_frames = []
        for frame in range(start, end+1):
            frame_boxes = vid_boxes[frame]
            if side not in frame_boxes or frame_boxes[side] is None:
                continue
            if cat not in frame_boxes or frame_boxes[cat] is None:
                continue
            valid_frames.append(frame)
        return valid_frames

    def __getitem__(self, index):
        """
        Returns:
            images: ndarray (N, H, W, 3) RGB
                note frankmocap requires `BGR` input
            hand_bbox_dicts: list of dict
                - left_hand/right_hand: ndarray (4,) of (x0, y0, w, h)
            obj_bbox_arrs: (N, 4) xywh
            object_masks: (N, H, W)
                - fg: 1, ignore -1, bg 0
            hand_masks: (N, H, W)
            cat: str, object categroy
        """
        if self.show_loading_time:
            _start_time = time.time()
        info = self.data_infos[index]
        vid, cat, visor_name, side, start, end = \
            info.vid, info.cat, info.visor_name, info.side, info.start, info.end

        valid_frames = self._keep_frame_with_boxes(vid, start, end, side, cat)
        if self.sample_frames < 0 and len(valid_frames) > 500:
            raise NotImplementedError(f"frames more than 500 : {len(valid_frames)}.")
        elif self.sample_frames < 0 or (len(valid_frames) < self.sample_frames):
            frames = valid_frames
        else:
            frame_indices = [v for v in np.linspace(0, len(valid_frames)-1, num=self.sample_frames, dtype=int)]
            frames = [valid_frames[i] for i in frame_indices]

        images = []
        hand_bbox_dicts = []
        obj_bbox_arrs = []
        object_masks = []
        hand_masks = []

        _side = 'left hand' if 'left' in side else 'right hand'
        side_id = self.cat_data_mapping[vid][_side]
        cid = self.cat_data_mapping[vid][visor_name]
        for frame_idx in frames:
            folder = self.locator.locate(vid, frame_idx)
            image = Image.open(self.image_fmt % (folder, vid, frame_idx))
            image = np.asarray(image)

            # bboxes
            hand_box = self._get_hand_box(vid, frame_idx, side)
            if side == 'right':
                hand_bbox_dict = dict(right_hand=hand_box, left_hand=None)
            elif side == 'left':
                hand_bbox_dict = dict(right_hand=None, left_hand=hand_box)
            else:
                raise ValueError(f"Unknown side {side}.")

            obj_bbox_arr = self._get_obj_box(vid, frame_idx, cat)

            # masks
            path = self.mask_fmt % (vid, vid, frame_idx)
            mask_hand, mask_obj = read_v3_mask_with_occlusion(
                path, side_id, cid,
                crop_hand_mask=self.crop_hand_mask,
                crop_hand_expand=HAND_MASK_KEEP_EXPAND,
                hand_box=self._get_hand_box(vid, frame_idx, side, expand=False),
                occlude_level=self.occlude_level)

            images.append(image)
            hand_bbox_dicts.append(hand_bbox_dict)
            obj_bbox_arrs.append(obj_bbox_arr)
            hand_masks.append(mask_hand)
            object_masks.append(mask_obj)

        side_return = f"{side}_hand"
        images = np.asarray(images)
        obj_bbox_arrs = torch.as_tensor(obj_bbox_arrs)
        hand_masks = torch.as_tensor(hand_masks)
        object_masks = torch.as_tensor(object_masks)

        global_cam = self._get_camera()
        batch_global_cam = global_cam.repeat(len(frames), device='cpu')

        element = DataElement(
            images=images,
            hand_bbox_dicts=hand_bbox_dicts,
            side=side_return,
            obj_bboxes=obj_bbox_arrs,
            hand_masks=hand_masks,
            object_masks=object_masks,
            cat=cat,
            global_camera=batch_global_cam
        )
        if self.show_loading_time:
            print(f"Data Loading time (s): {time.time() - _start_time}")
        return element


if __name__ == '__main__':
    # Debug dataset
    parser = ArgumentParser()
    parser.add_argument('index', type=int)
    parser.add_argument('--image_sets', type=str, default='/home/skynet/Zhifan/epic_analysis/hos/tools/model-input-Feb03.json')
    parser.add_argument('--sample_frames', type=int, default=30)
    args = parser.parse_args()

    dataset = EpicClipDatasetV3(
        image_sets=args.image_sets, sample_frames=args.sample_frames)
    element = dataset[args.index]