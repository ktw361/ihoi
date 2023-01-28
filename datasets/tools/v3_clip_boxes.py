import os
import gc
import numpy as np
import cv2

from argparse import ArgumentParser
from PIL import Image
from moviepy import editor
import bisect, re
from libzhifan import io
import tqdm
from config.epic_constants import HAND_MASK_KEEP_EXPAND, EPIC_HOA_SIZE, VISOR_SIZE
from datasets.epic_lib import epichoa
from datasets.epic_clip_v3 import PairLocator, ClipInfo

from libzhifan import odlib
odlib.setup('xywh')

"""
generates a cache
all_boxes: dict
- vid: dict
    - frame: dict
        - side: np.ndarray [4] or None
        - cat: np.ndarray [4] or None
    If a frame has been processed, all_boxes[vid][frame] must exist.
"""


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('input', type=str, help='Input file')
    parser.add_argument('--generate_videos', action='store_true', help='Generate video')
    parser.add_argument('--output', default='/home/skynet/Zhifan/ihoi/weights/v3_clip_boxes.pkl')
    parser.add_argument('--amend', action='store_true', help='Amend the output file instead of restarting')
    parser.add_argument('--index', default=None, type=int)
    args = parser.parse_args()
    return args


class Mapper:
    """ Mapping from VISOR frame to EPIC frame """

    def __init__(self, mapping):
        """
        Args:
            mapping: e.g. '/home/skynet/Zhifan/data/epic_analysis/resources/mapping_visor_to_epic.json'
        """
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


def generate_boxes(hos_v3,
                   output,
                   gen_videos=False,
                   amend=True,
                   index: int = None):
    hoa_root = '/home/skynet/Zhifan/datasets/epic/hoa/'
    mask_fmt = '/media/skynet/DATA/Datasets/visor-dense/interpolations/%s/%s_frame_%010d.png'  # % (vid, vid, frame)
    data_mapping = io.read_json('/media/skynet/DATA/Datasets/visor-dense/meta_infos/data_mapping.json')
    mapper = Mapper(mapping='/home/skynet/Zhifan/data/epic_analysis/resources/mapping_visor_to_epic.json')
    locator = PairLocator()
    image_fmt = '/media/skynet/DATA/Datasets/visor-dense/480p/%s/%s_frame_%010d.jpg'  # % (folder, vid, frame)

    hos_v3 = io.read_json(hos_v3)
    hos_v3 = [ClipInfo(**v) for v in hos_v3]
    hos_v3 = [v for v in hos_v3 if v.status == 'FOUND']
    if index is not None:
        hos_v3 = hos_v3[index:index+1]
    if amend:
        all_boxes = io.read_pickle(output)
    else:
        all_boxes = dict()

    os.makedirs('/home/skynet/Zhifan/ihoi/outputs/tmp/clip_boxes_videos', exist_ok=True)
    for clip in tqdm.tqdm(hos_v3):
        vid = clip.vid
        start = clip.start
        end = clip.end
        side = clip.side
        cat = clip.cat
        cid = data_mapping[vid][clip.visor_name]  # original name

        vid_boxes = all_boxes.get(vid, dict())
        frames = []

        if end < start:
            print('Error clip: ', clip)
            continue

        # If amend and all start-end frames are already computed, skip
        if amend:
            computed = True
            for frame in range(start, end+1):
                if not frame in vid_boxes:
                    computed = False
                    break
            if computed:
                continue

        df = epichoa.load_video_hoa(vid, hoa_root=hoa_root)

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
                f'/home/skynet/Zhifan/ihoi/outputs/tmp/clip_boxes_videos/{vid}_{start}_{end}.mp4', verbose=False, logger=None)
            del seq
            gc.collect()
        io.write_pickle(all_boxes, output)


if __name__ == '__main__':
    args = parse_args()
    generate_boxes(hos_v3=args.input, 
                   output=args.output,
                   gen_videos=args.generate_videos,
                   amend=args.amend,
                   index=args.index)