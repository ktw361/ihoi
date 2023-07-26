from typing import NamedTuple, List, Union
from argparse import ArgumentParser
import pickle, json
import os, re, bisect, time
from PIL import Image
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

from config.epic_constants import HAND_MASK_KEEP_EXPAND, EPIC_HOA_SIZE
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


class ClipInfo(NamedTuple):
    seqid: str
    vid: str
    cat: str
    hos_name: str
    side: str  # 'left' or 'right'
    st: int 
    ed: int


class DataElement(NamedTuple):
    images: list 
    hand_bbox_dicts: list 
    side: str 
    obj_bboxes: torch.Tensor
    hand_masks: torch.Tensor 
    object_masks: torch.Tensor 
    cat: str
    global_camera: BatchCameraManager


EPICHOR_HOCROP_SIZE = (256, 256)


class EPICHORDataset(Dataset):

    def __init__(self,
                 image_sets,
                 all_boxes='./epic_hor_data/detections/hocrop_clip_boxes.pkl',
                 image_size=EPICHOR_HOCROP_SIZE,
                 hand_expansion=0.4,
                 crop_hand_mask=True,
                 occlude_level='all',
                 sample_frames=30,
                 show_loading_time=False,
                 *args,
                 **kwargs):
        """_summary_

        Args:
            image_sets (str): path to seq files
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
        self.show_loading_time = show_loading_time
        self.occlude_level = occlude_level

        self.image_fmt = './epic_hor_data/hocrop_seqs/%s/frame_%010d.jpg'  # % (seqid, frame_idx)
        self.mask_fmt = './epic_hor_data/hocrop_mask_seqs/%s/frame_%010d.png'  # % (seqid, frame_idx)

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
            valid_ann_segments = json.load(fp)

        seqid_to_mp4 = io.read_json('./epic_hor_data/seqid_to_mp4.json')
        infos = []
        for seqid, mp4_name in seqid_to_mp4.items():
            if mp4_name not in valid_ann_segments:
                continue
            v = valid_ann_segments[mp4_name]
            clip_info = ClipInfo(
                seqid=seqid,
                vid=v['vid'], cat=v['cat'], hos_name=v['obj'], side=v['handside'].replace(
                    ' hand', ''), st=v['st'], ed=v['ed'])
            infos.append(clip_info)
        return infos

    def __len__(self):
        return len(self.data_infos)
    
    def locate_index_from_seqid(self, seqid):
        search = [i for i, v in enumerate(self.data_infos)
                  if v.seqid == seqid]
        return search[0]

    def _get_hand_box(self, vid, frame_idx, side, expand=True):
        w, h = self.image_size
        hand_box = self.ho_boxes[vid][frame_idx][side]  # xywh normed
        if not expand:
            return hand_box
        hand_box = hand_box * np.array([w, h, w, h])
        hand_box_xyxy = xywh_to_xyxy(hand_box)
        hand_box_squared_xyxy = square_bbox(
            hand_box_xyxy[None], pad=self.hand_expansion)[0]
        hand_box_squared_xyxy[:2] = hand_box_squared_xyxy[:2].clip(min=[0, 0])
        hand_box_squared_xyxy[2:] = hand_box_squared_xyxy[2:].clip(max=[w, h])
        hand_box_squared = xyxy_to_xywh(hand_box_squared_xyxy)
        return hand_box_squared

    def _get_obj_box(self, vid, frame_idx, cat):
        w, h = self.image_size
        return self.ho_boxes[vid][frame_idx][cat] * np.array([w, h, w, h])

    def visualize_bboxes(self, index):
        raise NotImplementedError
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
        img_h, img_w = self.image_size
        global_cam = CameraManager(
            fx=512, fy=512, cx=512, cy=0, img_h=img_h, img_w=img_w)
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
            if frame not in vid_boxes:
                continue
            frame_boxes = vid_boxes[frame]
            if side not in frame_boxes or frame_boxes[side] is None:
                continue
            if cat not in frame_boxes or frame_boxes[cat] is None:
                continue
            valid_frames.append(frame)
        return valid_frames
    
    def get_input_frame_indices(self, index):
        info = self.data_infos[index]
        vid, cat, hos_name, side, start, end = \
            info.vid, info.cat, info.hos_name, info.side, info.st, info.ed

        valid_frames = self._keep_frame_with_boxes(vid, start, end, side, cat)
        if self.sample_frames < 0 and len(valid_frames) > 500:
            raise NotImplementedError(f"frames more than 500 : {len(valid_frames)}.")
        elif self.sample_frames < 0 or (len(valid_frames) < self.sample_frames):
            frames = valid_frames
        else:
            frame_indices = [v for v in np.linspace(0, len(valid_frames)-1, num=self.sample_frames, dtype=int)]
            frames = [valid_frames[i] for i in frame_indices]
        return frames

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
        seqid = info.seqid
        vid, cat, hos_name, side, start, end = \
            info.vid, info.cat, info.hos_name, info.side, info.st, info.ed

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
        for frame_idx in frames:
            image = Image.open(self.image_fmt % (seqid, frame_idx))
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
            HAND_ID = 1
            OBJ_ID = 2
            mask_path = self.mask_fmt % (seqid, frame_idx)
            mask_hand, mask_obj = read_v3_mask_with_occlusion(
                mask_path, HAND_ID, OBJ_ID,
                crop_hand_mask=False, # self.crop_hand_mask,
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
    parser.add_argument('--image_sets', type=str, default='/media/barry/DATA/Zhifan/epic_hor_data/valid_mp4_segments.json')
    parser.add_argument('--sample_frames', type=int, default=30)
    args = parser.parse_args()

    dataset = EPICHORDataset(
        image_sets=args.image_sets, sample_frames=args.sample_frames)
    element = dataset[args.index]