from typing import NamedTuple
from PIL import Image
import numpy as np
import torch
from nnutils.image_utils import square_bbox
from libzhifan.odlib import xyxy_to_xywh, xywh_to_xyxy
from libzhifan.geometry import CameraManager, BatchCameraManager
from torch.utils.data import Dataset


class DataElement(NamedTuple):
    images: list 
    hand_bbox_dicts: list 
    side: str 
    obj_bboxes: torch.Tensor
    hand_masks: torch.Tensor 
    object_masks: torch.Tensor 
    cat: str
    global_camera: BatchCameraManager


class ArcticStableDataset(Dataset):

    def __init__(self):
        # sub/seq/view
        self.hand_expansion = 0.4
        self.image_size = (728, 520)
        self.data_infos = [
            ('s01/ketchup_grab_01/0', 'left', 'ketchup', 211, 237),
        ]

    def __len__(self):
        return len(self.data_infos)
    
    def _get_camera(self, index) -> CameraManager:
        K_ego = np.array([
            [2.4141504e+03, 0.0000000e+00, 1.3286525e+03],
            [0.0000000e+00, 2.4135042e+03, 9.8284296e+02],
            [0.0000000e+00, 0.0000000e+00, 1.0000000e+00]], dtype=np.float32)
        low_h = 520
        low_w = 728
        orig_h = 2000
        orig_w = 2800
        w_ratio = low_w / orig_w
        h_ratio = low_h / orig_h
        fx = K_ego[0, 0] * w_ratio
        fy = K_ego[1, 1] * h_ratio
        cx = K_ego[0, 2] * w_ratio
        cy = K_ego[1, 2] * h_ratio
        cam_manager = CameraManager(fx, fy, cx, cy, img_h=low_h, img_w=low_w)
        return cam_manager

    def _get_hand_box(self, bboxes, frame_idx, side, expand=True):
        hand_box = bboxes[side][frame_idx]
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

    def __getitem__(self, index) -> DataElement:
        info = self.data_infos[index]
        full_seq, side, cat, start, end = info

        image_format = f'arctic_data/images_low/{full_seq}/%05d.jpg'
        mask_format = f'arctic_outputs/masks_low/{full_seq}/%05d.png'

        images = []
        hand_bbox_dicts = []
        obj_bbox_arrs = []
        hand_masks = []
        object_masks = []
        bboxes = np.load(f'arctic_outputs/bboxes_low/{full_seq}/bboxes.npy', allow_pickle=True).item()
        for frame_idx in range(start, end+1):
            image = np.asarray(Image.open(image_format % frame_idx))
            mask = np.asarray(Image.open(mask_format % frame_idx))
            hand_box = self._get_hand_box(bboxes, frame_idx, side)
            bbox_o = bboxes['object'][frame_idx]
            if side == 'right':
                hand_bbox_dict = dict(right_hand=hand_box, left_hand=None)
            elif side == 'left':
                hand_bbox_dict = dict(right_hand=None, left_hand=hand_box)
            images.append(image)
            obj_bbox_arrs.append(bbox_o)
            hand_bbox_dicts.append(hand_bbox_dict)
            hand_mask = mask == (2 if side == 'left' else 3)
            obj_mask = mask == 1
            hand_masks.append(hand_mask)
            object_masks.append(obj_mask)

        side_return = side + '_hand'
        images = np.asarray(images)
        obj_bbox_arrs = torch.as_tensor(obj_bbox_arrs)
        hand_masks = torch.as_tensor(hand_masks)
        object_masks = torch.as_tensor(object_masks)
        global_cam = self._get_camera(index)
        batch_global_cam = global_cam.repeat(len(images), device='cpu')
        cat = 'arctic_' + cat
        element = DataElement(
            images=images, hand_bbox_dicts=hand_bbox_dicts, side=side_return,
            obj_bboxes=obj_bbox_arrs, hand_masks=hand_masks,
            object_masks=object_masks, cat=cat, global_camera=batch_global_cam)
        return element



if __name__ == '__main__':
    dataset = ArcticStableDataset()
    e = dataset[0]
    print("Done")

