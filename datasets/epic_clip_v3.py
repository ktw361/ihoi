from typing import NamedTuple, List, Union
import pickle, json
import os, re, bisect
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
    """ locate a (vid, frame) in P01_01_0003 """
    def __init__(self,
                 result_root='/home/skynet/Zhifan/data/visor-dense/480p',
                 pair_infos='/home/skynet/Zhifan/data/visor-dense/meta_infos/480p_pair_infos.txt',
                 verbose=True):
        self.result_root = Path(result_root)
        with open(pair_infos) as fp:
            pair_infos = fp.readlines()
            pair_infos = [v.strip().split(' ') for v in pair_infos]

        self._build_index(pair_infos)
        self.verbose = verbose

    def _build_index(self, pair_infos: list):
        """ pair_infos[i] = ['P01_01_0003', '123', '345']
        """
        self._all_full_frames = []
        self._all_folders = []
        for folder, st, ed in pair_infos:
            min_frame = int(st)
            index = self._hash(folder, min_frame)
            self._all_full_frames.append(index)
            self._all_folders.append(folder)

        self._all_full_frames = np.int64(self._all_full_frames)
        sort_idx = np.argsort(self._all_full_frames)
        self._all_full_frames = self._all_full_frames[sort_idx]
        self._all_folders = np.asarray(self._all_folders)[sort_idx]

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


class ClipInfo(NamedTuple):
    vid: str
    gt_frames: str
    cat: str
    visor_name: str
    st_bound: int
    ed_bound: int
    side: str
    start: int
    end: int
    comments: str


class DataElement(NamedTuple):
    images: list 
    hand_bbox_dicts: list 
    side_return: str 
    obj_bboxes: torch.Tensor
    hand_masks: torch.Tensor 
    object_masks: torch.Tensor 
    cat: str


class EpicClipDatasetV3(Dataset):

    def __init__(self,
                 image_sets='/home/skynet/Zhifan/htmls/hos_v3_react/hos_step5_in_progress.json',
                 all_boxes='/home/skynet/Zhifan/ihoi/weights/v3_clip_boxes.pkl',
                 cat_data_mapping='/media/skynet/DATA/Datasets/visor-dense/meta_infos/data_mapping.json',
                 image_size=VISOR_SIZE,
                 hand_expansion=0.4,
                 crop_hand_mask=True,
                 sample_frames=20,  # TODO, take into account mask quality?
                 *args,
                 **kwargs):
        """_summary_

        Args:
            image_sets (str): path to clean set frames
            image_size: Tuple of (W, H)
            hand_expansion (float): size of hand bounding box after squared.
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

        def is_valid(info: ClipInfo):
            if 'manip' in info.comments:
                return False
            if 'short' in info.comments:
                return False
            if 'del' in info.comments:
                return False
            return info.start != -1 and info.end != -1

        infos = [ClipInfo(**v) for v in infos]
        return list(filter(is_valid, infos))

    def __len__(self):
        return len(self.data_infos)

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

    def get_camera(self, index=-1) -> BatchCameraManager:
        global_cam = CameraManager(
            # fx=1050, fy=1050, cx=960, cy=540,
            fx=1050, fy=1050, cx=1280, cy=0,
            img_h=1080, img_w=1920)
        new_w, new_h = self.image_size
        global_cam = global_cam.resize(new_h=new_h, new_w=new_w)
        info = self.data_infos[index]
        bsize = info.end - info.start + 1
        if self.sample_frames > 0:
            bsize = min(self.sample_frames, bsize)
        batch_global_cam = global_cam.repeat(bsize, device='cpu')
        return batch_global_cam
    
    def _keep_frame_with_boxes(self, vid, start, end, side, cat) -> List[int]:
        """ 
        Returns:
            a list of frames in which both obj and hand box are present
        """
        vid_boxes = self.ho_boxes[vid]
        valid_frames = []
        for frame in range(start, end+1):
            if side not in vid_boxes[frame]:
                continue
            if cat not in vid_boxes[frame]:
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
        info = self.data_infos[index]
        vid, cat, visor_name, side, start, end = \
            info.vid, info.cat, info.visor_name, info.side, info.start, info.end

        valid_frames = self._keep_frame_with_boxes(vid, start, end, side, cat)
        if self.sample_frames < 0 and len(valid_frames) > 100:
            raise NotImplementedError(f"frames more than 100 : {len(valid_frames)}.")
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
                hand_box=self._get_hand_box(vid, frame_idx, side, expand=False))

            images.append(image)
            hand_bbox_dicts.append(hand_bbox_dict)
            obj_bbox_arrs.append(obj_bbox_arr)
            hand_masks.append(mask_hand)
            object_masks.append(mask_obj)

        side_return = f"{side}_hand"
        images = np.stack(images)
        obj_bbox_arrs = torch.as_tensor(obj_bbox_arrs)
        hand_masks = torch.as_tensor(hand_masks)
        object_masks = torch.as_tensor(object_masks)

        element = DataElement(
            images=images,
            hand_bbox_dicts=hand_bbox_dicts,
            side_return=side_return,
            obj_bboxes=obj_bbox_arrs,
            hand_masks=hand_masks,
            object_masks=object_masks,
            cat=cat
        )
        return element


def generate_boxes(hos_v3='/home/skynet/Zhifan/htmls/hos_v3_react/hos_step5_in_progress.json',
                   gen_videos=False):

    from PIL import Image
    from moviepy import editor
    import bisect, re
    from libzhifan import io
    import tqdm
    from datasets.epic_lib import epichoa

    class Mapper:
        """ Mapping from VISOR frame to EPIC frame """

        def __init__(self,
                    mapping='/home/skynet/Zhifan/data/epic_analysis/resources/mapping_visor_to_epic.json'):
            mapping = io.read_json(mapping)
            cvt = lambda x : int(re.search('\d{10}', x).group(0))
            vids = mapping.keys()

            self.visor = dict()
            self.epic = dict()
            for vid in vids:
                src, dst = list(zip(*mapping[vid].items()))
                src = list(map(cvt, src))
                dst = list(map(cvt, dst))
                self.visor[vid] = src
                self.epic[vid] = dst

        def __call__(self, vid, frame: int) -> int:
            i = bisect.bisect_right(self.visor[vid], frame)  # first i s.t visor[i] strictly greater than frame
            if i == 0:
                a, b = 0, self.visor[vid][i]
                p, q = 0, self.epic[vid][i]
            elif i == len(self.visor[vid]):
                a = self.visor[vid][i-1]
                p = self.epic[vid][i-1]
                return frame - a + p
            else:
                a, b = self.visor[vid][i-1], self.visor[vid][i]
                p, q = self.epic[vid][i-1], self.epic[vid][i]
            k = (frame - a) / (b - a)
            y = k * (q - p) + p
            return round(y)

    def row2xywh(row):
        wid = row.right - row.left
        hei = row.bottom - row.top
        return np.asarray([row.left, row.top, wid, hei])

    def box_dist(box1, box2):
        """ distance for xywh box (4,) """
        dist = np.linalg.norm(box1[:2] - box2[:2])
        return dist

    def compute_hand_box(det_box: np.ndarray,
                         mask: np.ndarray,
                         hid: int) -> np.ndarray:
        """ Compute hand_box using mask but with region inside expanded det_box only.
            1. Expand det_box
            2. Keep mask inside det_box only
            3. Compute box, which is tight

        Args:
            hand_box: (4,) xywh
            mask: (H, W) np.uint8
            hid: int. Hand index

        Returns:
            None if mask doesn't contain hid in det_box region
            box: (4,) xywh
        """
        H, W = mask.shape
        det_box = det_box.astype(int)
        x, y, bw, bh = det_box
        det_size = max(bw, bh)
        pad = det_size * HAND_MASK_KEEP_EXPAND / 2
        x0 = max(int(x - pad), 0)
        y0 = max(int(y - pad), 0)
        x1 = min(int(x + bw + pad), W-1)
        y1 = min(int(y + bh + pad), H-1)

        crop_mask = np.zeros_like(mask)
        crop_mask[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
        # assert n_cls == 2
        h, w = mask.shape[:2]
        y, x = np.nonzero(crop_mask == hid)
        if len(y) == 0 or len(x) == 0:
            return None
        pad = 1
        x0, x1 = max(x.min()-pad, 0), min(x.max()+pad, w-1)
        y0, y1 = max(y.min()-pad, 0), min(y.max()+pad, h-1)
        hand_box = np.asarray([x0, y0, x1-x0, y1-y0], dtype=np.float32)
        return hand_box

    def boxes_from_mask(mask: np.ndarray,
                        size_mul=1.0,
                        pad=2,
                        debug=False) -> np.ndarray:
        """ (H, W) -> (N,4)
        If distance between two boxes <
            `size_mul` x max(size_1, size_2),
            merge them.
        Returns:
            (N, 4) xywh. or (0, 4) if no object
        """
        n_cls, mask = cv2.connectedComponents(
            mask, connectivity=8)

        boxes = []
        for c in range(1, n_cls):
            h, w = mask.shape[:2]
            y, x = np.nonzero(mask == c)
            x0, x1 = max(x.min()-pad, 0), min(x.max()+pad, w-1)
            y0, y1 = max(y.min()-pad, 0), min(y.max()+pad, h-1)
            box = np.asarray([x0, y0, x1-x0, y1-y0], dtype=np.float32)
            boxes.append(box)

        if len(boxes) == 0:
            return np.empty((0, 4), dtype=np.float32)

        visit = set()
        finals = []
        for i in range(len(boxes)):
            if i in visit:
                continue
            box1 = boxes[i]
            bc1 = box1[:2] + box1[2:]/2
            sz1 = np.linalg.norm(box1[2:])
            for j in range(i+1, len(boxes)):
                if j in visit:
                    continue
                box2 = boxes[j]
                bc2 = box2[:2] + box2[2:]/2
                dist = box_dist(box1, box2)
                sz2 = np.linalg.norm(box2[2:])
                if debug:
                    print(dist, sz1, sz2)
                if dist < size_mul * max(sz1, sz2): # Merge
                    x0 = min(box1[0], box2[0])
                    y0 = min(box1[1], box2[1])
                    x1 = max(box1[0]+box1[2], box2[0]+box2[2])
                    y1 = max(box1[1]+box1[3], box2[1]+box2[3])
                    box1 = np.float32([x0, y0, x1-x0, y1-y0])
                    visit.add(j)
            finals.append(box1)

        return np.stack(finals)
    
    def compute_box_iou(box1: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """
        Args:
            box1: (1, 4) xywh
            boxes: (N, 4) xywh
        Returns:
            ious: (N,)
        """
        x1, y1, w1, h1 = box1.T
        x2, y2, w2, h2 = boxes.T
        x1, x2 = np.maximum(x1, x2), np.minimum(x1+w1, x2+w2)
        y1, y2 = np.maximum(y1, y2), np.minimum(y1+h1, y2+h2)
        inter = np.maximum(x2-x1, 0) * np.maximum(y2-y1, 0)
        union = w1*h1 + w2*h2 - inter
        ious = inter / union
        return ious

    hoa_root = '/home/skynet/Zhifan/datasets/epic/hoa/'
    mask_fmt = '/media/skynet/DATA/Datasets/visor-dense/interpolations/%s/%s_frame_%010d.png'  # % (vid, vid, frame)
    data_mapping = io.read_json('/media/skynet/DATA/Datasets/visor-dense/meta_infos/data_mapping.json')
    mapper = Mapper()
    locator = PairLocator()
    image_fmt = '/media/skynet/DATA/Datasets/visor-dense/480p/%s/%s_frame_%010d.jpg'  # % (folder, vid, frame)

    hos_v3 = io.read_json(hos_v3)
    def is_valid(info: ClipInfo):
        if 'manip' in info.comments:
            return False
        if 'short' in info.comments:
            return False
        if 'del' in info.comments:
            return False
        return info.start != -1 and info.end != -1
    hos_v3 = list(filter(is_valid, [ClipInfo(**v) for v in hos_v3]))
    all_boxes = dict()

    os.makedirs('/home/skynet/Zhifan/ihoi/outputs/clip_boxes_videos', exist_ok=True)
    for clip in tqdm.tqdm(hos_v3):
        vid = clip.vid
        start = clip.start
        end = clip.end
        side = clip.side
        cat = clip.cat
        cid = data_mapping[vid][clip.visor_name]  # original name
        # if os.path.exists(f'/home/skynet/Zhifan/ihoi/outputs/clip_boxes_videos/{vid}_{start}_{end}.mp4'):
        #     continue
        df = epichoa.load_video_hoa(vid, hoa_root=hoa_root)
        vid_boxes = all_boxes.get(vid, dict())
        frames = []
        for frame in range(start, end+1):
            mask = mask_fmt % (vid, vid, frame)
            mask = Image.open(mask).convert('P')
            mask = np.asarray(mask, dtype=np.uint8)
            obj_mask = np.where(mask == cid, cid, 0).astype(np.uint8)
            boxes = boxes_from_mask(obj_mask)
            epic_frame = mapper(vid, frame)  # mapping from visor frame to epic frame, b.c. hoa is in EPIC

            if len(boxes) == 0:  # No object mask -> the Model should skip this frame
                if gen_videos:
                    image = image_fmt % ( locator.locate(vid, frame), vid, frame)
                    image = Image.open(image)
                    frames.append(img)
                frame_boxes = vid_boxes.get(frame, dict())
                frame_boxes[side] = None
                frame_boxes[cat] = None
                vid_boxes[frame] = frame_boxes
                continue

            hand_entries = df[(df.frame == epic_frame) & (df.det_type == 'hand') & (df.side == side)]
            if len(hand_entries) == 0:  # Put None to hand_box to indicate model skip this frame
                if gen_videos:
                    image = image_fmt % ( locator.locate(vid, frame), vid, frame)
                    image = Image.open(image)
                    img = odlib.draw_bboxes_image_array(image, obj_box[None], color='purple')
                    frames.append(img)
                frame_boxes = vid_boxes.get(frame, dict())
                frame_boxes[side] = None
                frame_boxes[cat] = obj_box
                vid_boxes[frame] = frame_boxes
                continue

            det_hand_box = row2xywh(df[
                (df.frame == epic_frame) & (df.det_type == 'hand') & (df.side == side)].iloc[0])
            det_hand_box = det_hand_box / (EPIC_HOA_SIZE * 2) * (VISOR_SIZE * 2)
            hand_name = ('left hand' if 'left' in side else 'right hand')
            hid = data_mapping[vid][hand_name]
            hand_mask = np.where(mask == hid, hid, 0).astype(np.uint8)
            hand_box = compute_hand_box(det_hand_box, hand_mask, hid)
            hand_box = hand_box if hand_box is not None else det_hand_box  # Handle no hand_mask corner case

            """ Only keep the box that is closest to hand """
            det_obj_entries = df[(df.frame == epic_frame) & (df.det_type == 'object')]
            if len(det_obj_entries) == 1:
                det_obj_box = row2xywh(det_obj_entries.iloc[0])
                det_obj_box = det_obj_box / (EPIC_HOA_SIZE * 2) * (VISOR_SIZE * 2)
                ious = compute_box_iou(det_obj_box[None], boxes)
                min_idx = np.argmax(ious)
            else:
                min_dst = np.inf
                min_idx = 0
                for bi in range(len(boxes)):
                    dst = box_dist(boxes[bi], hand_box)
                    if box_dist(boxes[bi], hand_box) < min_dst:
                        min_idx = bi
                        min_dst = dst
            obj_box = boxes[min_idx]

            if gen_videos:
                image = image_fmt % ( locator.locate(vid, frame), vid, frame)
                image = Image.open(image)
                img = odlib.draw_bboxes_image_array(image, obj_box[None], color='purple')
                odlib.draw_bboxes_image(img, hand_box[None],
                                        color='green' if 'right' in side else 'red')
                frames.append(img)
            frame_boxes = vid_boxes.get(frame, dict())
            frame_boxes[side] = hand_box
            frame_boxes[cat] = obj_box
            vid_boxes[frame] = frame_boxes
        all_boxes[vid] = vid_boxes
        if gen_videos:
            seq = editor.ImageSequenceClip(list(map(np.asarray, frames)), fps=15)
            seq.write_videofile(
                f'/home/skynet/Zhifan/ihoi/outputs/clip_boxes_videos/{vid}_{start}_{end}.mp4', verbose=False, logger=None)
    io.write_pickle(all_boxes, '/home/skynet/Zhifan/ihoi/weights/v3_clip_boxes.pkl')


if __name__ == '__main__':
    generate_boxes()
