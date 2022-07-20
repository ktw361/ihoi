#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path
from PIL import Image
import numpy as np

from . import epichoa


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