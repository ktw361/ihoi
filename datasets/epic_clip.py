from typing import NamedTuple
import pickle
import json
import os.path as osp
import numpy as np
from PIL import Image

from torch.utils.data import Dataset

from datasets.epic_lib.epic_utils import read_epic_image


HAND_MASK_KEEP_EXPAND = 0.2

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
                 crop_hand_mask=True,
                 *args,
                 **kwargs):
        """_summary_

        Args:
            image_sets (str): path to clean set frames
            epic_root (str):
            hoa_root (str):
            mask_dir (str):
            image_size: Tuple of (W, H)
            crop_hand_mask: If True, will crop hand mask with only pixels
                inside hand_bbox.
        """
        super().__init__(*args, **kwargs)
        self.epic_rgb_root = osp.join(epic_root, 'rgb_root')
        self.mask_dir = mask_dir
        self.hoa_root = osp.join(epic_root, 'hoa')
        self.image_size = image_size
        self.crop_hand_mask = crop_hand_mask

        self.box_scale = np.asarray(image_size * 2) / ((1920, 1080) * 2)
        self.data_infos = self._read_image_sets(image_sets)
        with open(all_boxes, 'rb') as fp:
            self.ho_boxes = pickle.load(fp)

    def _read_image_sets(self, image_sets):
        """
        Returns:
            list of ClipInfo(vid, nid, frame_idx, cat, side, start, end)
        """
        with open(image_sets) as fp:
            infos = json.load(fp)

        infos = [ClipInfo(**v) 
                 for v in infos 
                 if (v['vid'], v['gt_frame']) not in self.wrong_set]
        return infos

    def __len__(self):
        return len(self.data_infos)

    def _get_hand_box(self, vid, frame_idx, side):
        return self.ho_boxes[vid][frame_idx][side]

    def _get_obj_box(self, vid, frame_idx, cat):
        return self.ho_boxes[vid][frame_idx][cat]

    def __getitem__(self, index):
        """
        Returns:
            images: ndarray (N, H, W, 3) RGB
                note frankmocap requires `BGR` input
            hand_bbox_dicts: list of dict
                - left_hand/right_hand: ndarray (4,) of (x1, y1, w, h)
            obj_bbox_arrs: (N, 4) xywh
            object_masks: (N, H, W)
                - fg: 1, ignore -1, bg 0
            hand_masks: (N, H, W)
            cat: str, object categroy
        """
        info = self.data_infos[index]
        vid, cat, side, start, end = \
            info.vid, info.cat, info.side, info.start, info.end
        if end - start > 100:
            raise NotImplementedError(f"frames more than 100 : {end - start}.")
        images = []
        hand_bbox_dicts = []
        obj_bbox_arrs = []
        object_masks = []
        hand_masks = []
        for frame_idx in range(start, end+1):
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
            mask = Image.open(path).convert('P')
            mask = mask.resize(self.image_size, Image.NEAREST)
            mask = np.asarray(mask, dtype=np.float32)
            mask_hand = np.zeros_like(mask)
            mask_obj = np.zeros_like(mask)
            side_name = f"{side} hand"
            mask_hand[mask == epic_cats.index(side_name)] = 1
            mask_obj[mask == epic_cats.index(cat)] = 1
            if self.crop_hand_mask:
                crop_expand = HAND_MASK_KEEP_EXPAND
                x0, y0, bw, bh = hand_box
                x1, y1 = x0 + bw, y0 + bh
                x0 -= bw * crop_expand / 2
                y0 -= bh * crop_expand / 2
                x1 += bw * crop_expand / 2
                y1 += bh * crop_expand / 2
                x0, y0, x1, y1 = map(int, (x0, y0, x1, y1))
                x0 = min(max(0, x0), mask.shape[1])
                y0 = min(max(0, y0), mask.shape[0])
                x1 = min(max(0, x1), mask.shape[1])
                y1 = min(max(0, y1), mask.shape[0])
                mask_hand_crop = np.zeros_like(mask_hand)
                mask_hand_crop[y0:y1, x0:x1] = mask_hand[y0:y1, x0:x1]
                mask_hand = mask_hand_crop

            images.append(image)
            hand_bbox_dicts.append(hand_bbox_dict)
            obj_bbox_arrs.append(obj_bbox_arr)
            hand_masks.append(mask_hand)
            object_masks.append(mask_obj)

        side_return = f"{side}_hand"
        images = np.stack(images)
        obj_bbox_arrs = np.stack(obj_bbox_arrs)
        hand_masks = np.stack(hand_masks)
        object_masks = np.stack(object_masks)

        return images, hand_bbox_dicts, side_return, obj_bbox_arrs, hand_masks, object_masks, cat


class CachedEpicClipDataset(EpicClipDataset):

    def __init__(self,
                 *args,
                 **kwargs):
        """_summary_

        Args:
            image_sets (str): path to clean set frames
            epic_root (str):
            hoa_root (str):
            mask_dir (str):
            image_size: Tuple of (W, H)
            crop_hand_mask: If True, will crop hand mask with only pixels
                inside hand_bbox.
        """
        super().__init__(*args, **kwargs)
        self.cache_dir = 'output/epic_clip_cache'
    
    def __getitem__(self, index):
        with open(f'{self.cache_dir}/{index}.pkl', 'rb') as fp:
            return pickle.load(fp)


if __name__ == '__main__':
    dataset = EpicClipDataset(
        image_sets='/home/skynet/Zhifan/data/epic_analysis/gt_clips.json')
    item = (dataset[0])
    print(item)
