#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Tuple
from PIL import Image
import numpy as np


EPIC_CATS = [
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


def read_epic_image(video_id, 
                    frame_idx, 
                    root='/home/skynet/Zhifan/data/epic_analysis/visor_frames/', 
                    as_pil=False):
    """ Read VISOR image """
    root = Path(root)
    frame = root/video_id/f"frame_{frame_idx:010d}.jpg"
    frame = Image.open(frame)
    if as_pil:
        return frame
    return np.asarray(frame)


def read_epic_image_old(video_id, 
                        frame_idx, 
                        root='/home/skynet/Zhifan/data/epic_rgb_frames/', 
                        as_pil=False):
    """ Read Epic-kitchens image """

    root = Path(root)
    frame = root/video_id[:3]/video_id/f"frame_{frame_idx:010d}.jpg"
    frame = Image.open(frame)
    if as_pil:
        return frame
    return np.asarray(frame)


def read_mask_with_occlusion(path: str,
                             out_size: Tuple,
                             side: str,
                             cat: str,
                             crop_hand_mask=False,
                             crop_hand_expand=0.0,
                             hand_box: np.ndarray = None):
    """
    Args:
        path: <path_to_frame_xxx.png>
        out_size: output mask size [W, H]
        side: hand side, 'left' or 'right'
        cat: object category
        crop_hand_mask: whether to crop hand mask to hand_box region
        crop_hand_expand: expand ratio
        hand_box: (4,)
    
    Returns:
        mask_hand, mask_obj: np.ndarray (H, W) int32
            1 fg, -1 ignore, 0 bg
    """
    mask = Image.open(path).convert('P')
    mask = mask.resize(out_size, Image.NEAREST)
    # mask = np.asarray(mask, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.int32)
    mask_hand = np.zeros_like(mask)
    mask_obj = np.zeros_like(mask)
    if 'left' in side:
        side_name = "left hand"
    elif 'right' in side:
        side_name = "right hand"
    else:
        raise ValueError
    mask_hand[mask == EPIC_CATS.index(side_name)] = 1
    mask_obj[mask == EPIC_CATS.index(cat)] = 1
    if crop_hand_mask:
        x0, y0, bw, bh = hand_box
        x1, y1 = x0 + bw, y0 + bh
        x0 -= bw * crop_hand_expand / 2
        y0 -= bh * crop_hand_expand / 2
        x1 += bw * crop_hand_expand / 2
        y1 += bh * crop_hand_expand / 2
        x0, y0, x1, y1 = map(int, (x0, y0, x1, y1))
        x0 = min(max(0, x0), mask.shape[1])
        y0 = min(max(0, y0), mask.shape[0])
        x1 = min(max(0, x1), mask.shape[1])
        y1 = min(max(0, y1), mask.shape[0])
        mask_hand_crop = np.zeros_like(mask_hand)
        mask_hand_crop[mask_hand == 1] = -1
        mask_hand_crop[y0:y1, x0:x1] = mask_hand[y0:y1, x0:x1]
        mask_hand_crop[mask_obj == 1] = -1
        mask_hand = mask_hand_crop
    # This has to happen after cropping
    mask_obj[mask_hand == 1] = -1
    return mask_hand, mask_obj


def read_v3_mask_with_occlusion(path: str,
                                side_id: int,
                                cid: int,
                                crop_hand_mask=False,
                                crop_hand_expand=0.0,
                                hand_box: np.ndarray = None,
                                occlude_level='all'):
    """ For HOS_V3 mask
    Args:
        path: <path_to_frame_xxx.png>
        side: hand side index in data_mapping.json
        cid: object category index in data_mapping.json
        crop_hand_mask: whether to crop hand mask to hand_box region
        crop_hand_expand: expand ratio
        hand_box: (4,)
        occlude_level: 
            -all: all other things will be -1, instead of 0
            -ho: same as HOMan, only hand-object mutual occlusion will be -1,
                other occluders will be 0
            -none: no occlusion
    
    Returns:
        mask_hand, mask_obj: np.ndarray (H, W) int32
            1 fg, -1 ignore, 0 bg
    """
    mask = Image.open(path).convert('P')
    mask = np.asarray(mask, dtype=np.int32)
    mask_hand = np.zeros_like(mask)
    mask_obj = np.zeros_like(mask)
    mask_hand[mask == side_id] = 1
    if crop_hand_mask:
        x0, y0, bw, bh = hand_box
        x1, y1 = x0 + bw, y0 + bh
        x0 -= bw * crop_hand_expand / 2
        y0 -= bh * crop_hand_expand / 2
        x1 += bw * crop_hand_expand / 2
        y1 += bh * crop_hand_expand / 2
        x0, y0, x1, y1 = map(int, (x0, y0, x1, y1))
        x0 = min(max(0, x0), mask.shape[1])
        y0 = min(max(0, y0), mask.shape[0])
        x1 = min(max(0, x1), mask.shape[1])
        y1 = min(max(0, y1), mask.shape[0])
        mask_hand_crop = np.zeros_like(mask_hand)
        mask_hand_crop[mask_hand == 1] = -1
        mask_hand_crop[y0:y1, x0:x1] = mask_hand[y0:y1, x0:x1]
        mask_hand_crop[mask_obj == 1] = -1
        mask_hand = mask_hand_crop
    # This has to happen after cropping
    if occlude_level == 'all':
        unique_ids = np.unique(np.asarray(mask))
        for c in unique_ids:
            if c == 0:  # bg
                continue
            mask_obj[mask == c] = -1
            if c != side_id:
                mask_hand[mask == c] = -1
    elif occlude_level == 'ho':
        mask_obj[mask_hand == 1] = -1
        mask_hand[mask_obj == 1] = -1
    else:
        pass
    mask_obj[mask == cid] = 1
    # mask_hand[mask_hand == side_id] = -1
        
    return mask_hand, mask_obj


""" Hoa """
HOA_ROOT = '/home/skynet/Zhifan/datasets/epic/hoa/'
hoa_map = dict()


def convert_hoa_boxes(hoa_df,
                      frame=None,
                      det_type=None,
                      src_wh=(1920, 1080),
                      dst_wh=(456, 256)):
    """

    Args:
        hoa_df (DataFrame):
        frame (int): _description_
        det_type (str): one of {'object', 'hand'}

    Returns:
        bboxes: ndarray (N, 4) in (x1, y1, x2, y2)
    """
    sw, sh = src_wh
    dw, dh = dst_wh
    scale = np.asarray([dw, dh, dw, dh]) / [sw, sh, sw, sh]
    entries = hoa_df
    if frame is not None:
        entries = hoa_df[entries.frame == frame]
    if det_type is not None:
        entries = entries[entries.det_type == det_type]
    boxes = entries[['left', 'top', 'right', 'bottom']].to_numpy()
    boxes = boxes * scale
    return boxes