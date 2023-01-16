from typing import NamedTuple, List
import pickle
import json
import os.path as osp
import numpy as np
import torch

from torch.utils.data import Dataset

from config.epic_constants import HAND_MASK_KEEP_EXPAND
from nnutils.image_utils import square_bbox
from datasets.epic_lib.epic_utils import (
    read_epic_image, read_mask_with_occlusion)

from libzhifan.odlib import xyxy_to_xywh, xywh_to_xyxy
from libzhifan.geometry import CameraManager, BatchCameraManager

# Visualization
import matplotlib.pyplot as plt
from libzhifan import odlib
odlib.setup('xywh')
import cv2


epic_cats = [
    '_bg',
    'left hand',
    'right hand',
    'can',
    'cup',
    'plate',
    'bottle',
    'mug',
    'bowl',
]


def row2xywh(row):
    wid = row.right - row.left
    hei = row.bottom - row.top
    return np.asarray([row.left, row.top, wid, hei])


class ClipInfo(NamedTuple):
    vid: str
    gt_frame: str
    cat: str
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


class EpicClipDataset(Dataset):

    wrong_set = {
        # Hand box missing/wrong-side in hoa
        ('P04_13', 10440),
        ('P11_16', 18079),
        ('P12_04', 1828),
        ('P12_101', 21783),
        ('P15_02', 25465),
        ('P22_107', 6292),
        ('P28_109', 10026),
        ('P37_101', 70106),
        # object box no merged
        ('P01_103', 538),
        ('P03_04', 43470),
        ('P11_16', 16029),
        ('P37_101', 15996),
    }

    def __init__(self,
                 image_sets='/home/skynet/Zhifan/data/epic_analysis/gt_clips.json',
                 epic_root='/home/skynet/Zhifan/datasets/epic',
                 mask_dir='/home/skynet/Zhifan/data/epic_analysis/InterpV2',
                 all_boxes='/home/skynet/Zhifan/data/epic_analysis/clip_boxes.pkl',
                 image_size=(1280, 720), # (640, 360),
                 hand_expansion=0.4,
                 crop_hand_mask=True,
                 sample_frames=20,
                 *args,
                 **kwargs):
        """_summary_

        Args:
            image_sets (str): path to clean set frames
            epic_root (str):
            hoa_root (str):
            mask_dir (str):
            image_size: Tuple of (W, H)
            hand_expansion (float): size of hand bounding box after squared.
            crop_hand_mask: If True, will crop hand mask with only pixels
                inside hand_bbox.
            sample_frames (int):
                If clip has frames more than sample_frames,
                subsample them to a reduced number.
        """
        super().__init__(*args, **kwargs)
        self.epic_rgb_root = osp.join(epic_root, 'rgb_root')
        self.mask_dir = mask_dir
        self.hoa_root = osp.join(epic_root, 'hoa')
        self.image_size = image_size
        self.hand_expansion = hand_expansion
        self.crop_hand_mask = crop_hand_mask
        self.sample_frames = sample_frames

        self.box_scale = np.asarray(image_size * 2) / ((1920, 1080) * 2)
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

        infos = [ClipInfo(**v)
                 for v in infos
                 if (v['vid'], v['gt_frame']) not in self.wrong_set]
        infos = [v for v in infos if len(v.comments) == 0]
        return infos

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
        vid, cat, side, start, end = \
            info.vid, info.cat, info.side, info.start, info.end
        if self.sample_frames < 0 and end - start > 100:
            raise NotImplementedError(f"frames more than 100 : {end - start}.")
        images = []
        hand_bbox_dicts = []
        obj_bbox_arrs = []
        object_masks = []
        hand_masks = []
        if self.sample_frames < 0 or (end-start+1 < self.sample_frames):
            frames = range(start, end+1)
        else:
            frames = np.linspace(start, end, num=self.sample_frames, dtype=int)

        for frame_idx in frames:
            image = read_epic_image(
                vid, frame_idx, as_pil=True)
            image = image.resize(self.image_size)
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
            path = f'{self.mask_dir}/{vid}/frame_{frame_idx:010d}.png'
            mask_hand, mask_obj = read_mask_with_occlusion(
                path,
                out_size=self.image_size, side=side, cat=cat,
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


def generate_boxes():
    """ This script dumps:
    object boxes from interpolated masks;
    hand boxes from HOA datasets.
    """

    import sys
    from collections import namedtuple
    from pathlib import Path
    import tqdm
    import re
    import bisect
    import numpy as np
    from PIL import Image
    import cv2
    from moviepy import editor

    from libzhifan import io, odlib
    odlib.setup('xywh')
    import os
    os.chdir('/home/skynet/Zhifan/ihoi')
    from datasets.epic_lib import epichoa

    epic_cats = [
        '_bg',
        'left hand',
        'right hand',
        'can',
        'cup',
        'plate',
        'bottle',
        'mug',
        'bowl',
    ]

    HAND_EXPANSION = 0.2

    class Mapper:
        """ Mapping from VISOR frame to EPIC frame """

        def __init__(self,
                    mapping='/home/skynet/Zhifan/data/epic_analysis/mapping_visor_to_epic.json'):
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


    def box_dist(box1, box2):
        """ distance for xywh box (4,) """
        dist = np.linalg.norm(box1[:2] - box2[:2])
        return dist


    def row2xywh(row, image_size=(1280, 720)):
        wid = row.right - row.left
        hei = row.bottom - row.top
        box_scale = np.asarray(image_size * 2) / ((1920, 1080) * 2)
        return np.asarray([row.left, row.top, wid, hei]) * box_scale


    def boxes_from_mask(mask: np.ndarray,
                        size_mul=1.0,
                        pad=2,
                        debug=False) -> np.ndarray:
        """ (H, W) -> (N,4)
        If distance between two boxes <
            `size_mul` x max(size_1, size_2),
            merge them.
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


    def compute_hand_box(det_box: np.ndarray,
                        mask: np.ndarray,
                        hid: int) -> np.ndarray:
        """ Compute hand_box using mask but with region inside expanded det_box only.
            1. Expand det_box
            2. Keep mask inside det_box only
            3. Compute box, which is tight

        Args:
            hand_box: (4,)
            mask: (H, W) np.uint8
            hid: int. Hand index in epic_cats

        Returns:
            box: (4,)
        """
        H, W = mask.shape
        det_box = det_box.astype(int)
        x, y, bw, bh = det_box
        det_size = max(bw, bh)
        pad = det_size * HAND_EXPANSION / 2
        x0 = max(int(x - pad), 0)
        y0 = max(int(y - pad), 0)
        x1 = min(int(x + bw + pad), W-1)
        y1 = min(int(y + bh + pad), H-1)

        crop_mask = np.zeros_like(mask)
        crop_mask[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
        # assert n_cls == 2
        h, w = mask.shape[:2]
        y, x = np.nonzero(crop_mask == hid)
        pad = 1
        x0, x1 = max(x.min()-pad, 0), min(x.max()+pad, w-1)
        y0, y1 = max(y.min()-pad, 0), min(y.max()+pad, h-1)
        hand_box = np.asarray([x0, y0, x1-x0, y1-y0], dtype=np.float32)
        return hand_box

    gen_videos = False

    ClipInfo = namedtuple("ClipInfo", "vid gt_frame cat side start end comments")
    image_size = (1280, 720)

    hoa_root = '/home/skynet/Zhifan/datasets/epic/hoa/'
    mask_root = Path('/home/skynet/Zhifan/data/epic_analysis/InterpV2/')
    images_root = Path('/home/skynet/Zhifan/data/epic_analysis/visor_frames/')


    gt_clips = io.read_json('/home/skynet/Zhifan/data/epic_analysis/gt_clips.json')
    gt_clips = [ClipInfo(**v) for v in gt_clips if len(v['comments']) == 0]
    mapper = Mapper()

    wrong_set = {
        # Hand box missing/wrong-side in hoa
        ('P04_13', 10440),
        ('P11_16', 18079),
        ('P12_04', 1828),
        ('P12_101', 21783),
        ('P15_02', 25465),
        ('P22_107', 6292),
        ('P28_109', 10026),
        ('P37_101', 70106),
        # object box no merged
        ('P01_103', 538),
        ('P03_04', 43470),
        ('P11_16', 16029),
        ('P37_101', 15996),
    }

    all_boxes = dict()


    # (epic_frame, side)
    correct_hoa = {
            ('P12_04', 88660, 'right'): np.asarray([623.15788269, 258.75      ,  78.59649658, 109.6875    ])
            }
        
    gen_videos = True

    # All videos
    for clip in tqdm.tqdm(gt_clips):
        if (clip.vid, clip.gt_frame) in wrong_set:
            continue
        vid = clip.vid
        gt_frame = clip.gt_frame
        start = clip.start
        end = clip.end
        side = clip.side
        cat = clip.cat
        cid = epic_cats.index(cat)
        df = epichoa.load_video_hoa(vid, hoa_root=hoa_root)

        vid_boxes = all_boxes.get(vid, dict())

        frames = []
        for frame in range(start, end+1):
            fname = f"frame_{frame:010d}"
            mask = Image.open(mask_root/vid/f'{fname}.png').convert('P').resize(image_size)
            mask = np.asarray(mask, dtype=np.uint8)
            obj_mask = np.where(mask == cid, cid, 0).astype(np.uint8)
            boxes = boxes_from_mask(obj_mask)
            epic_frame = mapper(vid, frame)

            # Update hand_box
            if (vid, epic_frame, side) in correct_hoa:
                hand_box = correct_hoa[(epic_frame, side)]
            else:
                hand_box = row2xywh(df[(df.frame == epic_frame) & (df.det_type == 'hand') & (df.side == side)].iloc[0])
            hid = epic_cats.index('left hand' if 'left' in side else 'right hand')
            hand_mask = np.where(mask == hid, hid, 0).astype(np.uint8)
            hand_box = compute_hand_box(hand_box, hand_mask, hid)

            """ Only keep the box that is closest to hand """
            min_dst = np.inf
            min_idx = 0
            for bi in range(len(boxes)):
                dst = box_dist(boxes[bi], hand_box)
                if box_dist(boxes[bi], hand_box) < min_dst:
                    min_idx = bi
                    min_dst = dst
            obj_box = boxes[min_idx]
            fname = f"frame_{frame:010d}"
            if gen_videos:
                image = Image.open(images_root/vid/f'{fname}.jpg').resize(image_size)
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
            seq.write_videofile(f'/home/skynet/Zhifan/ihoi/output/tmp/{vid}_{gt_frame}.mp4', verbose=False, logger=None)
    io.write_pickle(all_boxes, '/home/skynet/Zhifan/data/epic_analysis/clip_boxes.pkl')


if __name__ == '__main__':
    dataset = EpicClipDataset(
        image_sets='/home/skynet/Zhifan/data/epic_analysis/gt_clips.json')
    item = (dataset[0])
    print(item)
