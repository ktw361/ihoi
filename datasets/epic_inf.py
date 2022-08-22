import os.path as osp
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from config.epic_constants import HAND_MASK_KEEP_EXPAND
from datasets.epic_lib.epic_utils import (
    read_epic_image, read_mask_with_occlusion)
from datasets.epic_lib import epichoa

from libzhifan.geometry import CameraManager

""" Epic-Kitchens Inference Dataset """


def row2xywh(row):
    wid = row.right - row.left
    hei = row.bottom - row.top
    return np.asarray([row.left, row.top, wid, hei])


class EpicInference(Dataset):

    def __init__(self,
                 image_sets='/home/skynet/Zhifan/data/epic_analysis/interpolation',
                 epic_root='/home/skynet/Zhifan/datasets/epic',
                 epic_rgb_root='/home/skynet/Zhifan/data/epic_analysis/visor_frames/',
                 mask_dir='/home/skynet/Zhifan/data/epic_analysis/InterpV2',
                 image_size=(1280, 720), # (640, 360),
                 crop_hand_mask=False,
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
        self.epic_rgb_root = epic_rgb_root
        self.mask_dir = mask_dir
        self.hoa_root = osp.join(epic_root, 'hoa')
        self.image_size = image_size
        self.crop_hand_mask = crop_hand_mask

        self.box_scale = np.asarray(image_size * 2) / ((1920, 1080) * 2)
        self.data_infos = self._read_image_sets(image_sets)
        self.hoa_df = dict()

    def _read_image_sets(self, image_sets):
        """ 
        Returns:
            list with (vid, nid, frame_idx, cat, side)
        """
        with open(image_sets) as fp:
            lines = fp.readlines()

        data_infos = []
        for i, line in enumerate(lines):
            line = line.strip().replace('\t', ' ')
            nid, cat, side, st_frame = [
                v for v in line.split(' ') if len(v) > 0]
            vid = '_'.join(nid.split('_')[:2])
            st_frame = int(st_frame)
            data_infos.append((
                vid, nid, st_frame, cat, side))
        
        return data_infos

    def __len__(self):
        return len(self.data_infos)

    def _get_hand_entries(self, vid, frame_idx):
        if vid in self.hoa_df:
            vid_df = self.hoa_df[vid]
        else:
            vid_df = epichoa.load_video_hoa(vid, self.hoa_root)
            self.hoa_df[vid] = vid_df
        entries = vid_df[vid_df.frame == frame_idx]
        entries = entries[entries.det_type == 'hand']
        return entries

    def _get_obj_entries(self, vid, frame_idx):
        if vid in self.hoa_df:
            vid_df = self.hoa_df[vid]
        else:
            vid_df = epichoa.load_video_hoa(vid, self.hoa_root)
            self.hoa_df[vid] = vid_df
        entries = vid_df[vid_df.frame == frame_idx]
        entries = entries[entries.det_type == 'object']
        return entries

    def get_vid_frame(self, index):
        """ 
        Given an index, we also want to know which vid & frame 
        will be extracted by __getitem__
        """
        vid, _, frame_idx, _, _ = self.data_infos[index]
        return vid, frame_idx

    def get_camera(self, index=-1) -> CameraManager:
        cam_global = CameraManager(
            # fx=1050, fy=1050, cx=960, cy=540,
            fx=1050, fy=1050, cx=1280, cy=0,
            img_h=1080, img_w=1920)
        new_w, new_h = self.image_size
        return cam_global.resize(new_h=new_h, new_w=new_w)

    def __getitem__(self, index):
        """ 
        Returns:
            image: ndarray (H, W, 3) RGB
                note frankmocap requires `BGR` input
                image is in numpy format as it doesn't participate optimization
            hand_bbox_dict: dict
                - left_hand/right_hand: ndarray (4,) of (x1, y1, w, h)
            obj_bbox_ten: torch.Tensor (4,) xywh
            object_mask: torch.Tensor (H, W) 
                - fg: 1, ignore -1, bg 0 
            cat: str, object categroy
        """
        # image
        vid, _, frame_idx, cat, side = self.data_infos[index]
        image = read_epic_image(
            vid, frame_idx, root=self.epic_rgb_root, as_pil=True)
        image = image.resize(self.image_size)
        image = np.asarray(image)

        # bboxes
        hand_entries = self._get_hand_entries(vid, frame_idx)
        hand_entry = hand_entries[hand_entries.side == side].iloc[0]
        hand_box = row2xywh(hand_entry)
        hand_box = hand_box * self.box_scale
        if side == 'right':
            hand_bbox_dict = dict(right_hand=hand_box, left_hand=None)
        elif side == 'left':
            hand_bbox_dict = dict(right_hand=None, left_hand=hand_box)
        else:
            raise ValueError(f"Unknown side {side}.")

        obj_entries = self._get_obj_entries(vid, frame_idx)
        obj_entry = obj_entries.iloc[0]
        obj_bbox_arr = (row2xywh(obj_entry) * self.box_scale).astype(np.float32)

        # masks
        path = f'{self.mask_dir}/{vid}/frame_{frame_idx:010d}.png'
        mask_hand, mask_obj = read_mask_with_occlusion(
            path, 
            out_size=self.image_size, side=side, cat=cat, 
            crop_hand_mask=self.crop_hand_mask, 
            crop_hand_expand=HAND_MASK_KEEP_EXPAND,
            hand_box=hand_box)

        side = f"{side}_hand"
        obj_bbox_ten = torch.as_tensor(obj_bbox_arr)
        mask_hand = torch.as_tensor(mask_hand)
        mask_obj = torch.as_tensor(mask_obj)

        return image, hand_bbox_dict, side, obj_bbox_ten, \
            mask_hand, mask_obj, cat
    

if __name__ == '__main__':
    dataset = EpicInference(
        image_sets='/home/skynet/Zhifan/data/epic_analysis/clean_frame_debug.txt')
    print(dataset[0])