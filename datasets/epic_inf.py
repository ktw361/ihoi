import os.path as osp
import numpy as np
from PIL import Image

from torch.utils.data import Dataset

from datasets.epic_lib.epic_utils import read_epic_image
from datasets.epic_lib import epichoa


""" Epic-Kitchens Inference Datasetf """


def row2xywh(row):
    wid = row.right - row.left
    hei = row.bottom - row.top
    return np.asarray([row.left, row.top, wid, hei])


class EpicInference(Dataset):

    def __init__(self,
                 image_sets='/home/skynet/Zhifan/data/epic_analysis/interpolation',
                 epic_root='/home/skynet/Zhifan/datasets/epic',
                 mask_dir='/home/skynet/Zhifan/data/epic_analysis/interpolation',
                 image_size=(640, 360),
                 *args,
                 **kwargs):
        """_summary_

        Args:
            image_sets (str): path to clean set frames
            epic_root (str): 
            hoa_root (str): 
            mask_dir (str): 
            image_size: Tuple of (W, H)
        """
        super(EpicInference, self).__init__(*args, **kwargs)
        self.epic_rgb_root = osp.join(epic_root, 'rgb_root')
        self.mask_dir = mask_dir
        self.hoa_root = osp.join(epic_root, 'hoa')
        self.image_size = image_size

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
            # side = '_'.join([side, 'hand'])
            # TODO support both sides
            if side == 'left':
                continue
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

    def get_vid_frame(self, index):
        """ 
        Given an index, we also want to know which vid & frame 
        will be extracted by __getitem__
        """
        vid, nid, frame_idx, cat, side = self.data_infos[index]
        return vid, frame_idx

    def __getitem__(self, index):
        """ 
        Returns:
            image: ndarray (H, W, 3) RGB
                note frankmocap requires `BGR` input
            hand_bbox_list: list of dict, len 1
                - left_hand/right_hand: ndarray (4,) of (x1, y1, w, h)
            object_mask: (H, W)
        """
        vid, nid, frame_idx, cat, side = self.data_infos[index]
        image = read_epic_image(
            vid, frame_idx, root=self.epic_rgb_root, as_pil=True)
        image = image.resize(self.image_size)
        image = np.asarray(image)

        path = f'{self.mask_dir}/{vid}/frame_{frame_idx:010d}.png'
        mask = Image.open(path).convert('P')
        mask = mask.resize(self.image_size, Image.NEAREST)
        mask = np.asarray(mask, dtype=np.uint8)
        # TODO: object mask?

        hand_entries = self._get_hand_entries(vid, frame_idx)
        hand_entry = hand_entries[hand_entries.side == side].iloc[0]
        r_hand = row2xywh(hand_entry)
        r_hand = r_hand * self.box_scale
        hand_bbox_list = [dict(right_hand=r_hand, left_hand=None)]

        return image, hand_bbox_list, mask
    

# if __name__ == '__main__':
#     dataset = EpicInference(
#         image_sets='/home/skynet/Zhifan/data/epic_analysis/clean_frame_debug.txt')
#     print(dataset[0])