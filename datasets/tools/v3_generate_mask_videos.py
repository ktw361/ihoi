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


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('input', type=str, help='Input file')
    parser.add_argument('--sample_frames', type=int, default=30)
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


def generate_boxes(hos_v3,
                    sample_frames=30,
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

    videos_dir = '/home/skynet/Zhifan/ihoi/outputs/tmp/v3_mask_videos'
    os.makedirs(videos_dir, exist_ok=True)
    for clip in tqdm.tqdm(hos_v3):
        vid = clip.vid
        start = clip.start
        end = clip.end
        side = clip.side
        cat = clip.cat
        cid = data_mapping[vid][clip.visor_name]  # original name

        if end < start:
            print('Error clip: ', clip)
            continue

        frames = []
        frame_range = np.linspace(start, end, sample_frames, dtype=np.int)
        for frame in frame_range:
            image = image_fmt % ( locator.locate(vid, frame), vid, frame)
            image = Image.open(image)

            mask = mask_fmt % (vid, vid, frame)
            mask = Image.open(mask)
            mask_int = np.asarray(mask.convert('P'), dtype=np.uint8)
            mask_rgb = np.asarray(mask.convert('RGB'))
            # obj_mask = np.where(mask == cid, cid, 0).astype(np.uint8)
            hand_name = ('left hand' if 'left' in side else 'right hand')
            hid = data_mapping[vid][hand_name]
            canvas = np.zeros_like(mask_rgb)
            canvas[mask_int==cid, ...] = mask_rgb[mask_int==cid, ...]
            canvas[mask_int==hid, ...] = mask_rgb[mask_int==hid, ...]

            frame = np.vstack([image, canvas])
            frames.append(frame)

        seq = editor.ImageSequenceClip(list(map(np.asarray, frames)), fps=5)
        seq.write_videofile(
            f'{videos_dir}/{vid}_{start}_{end}.mp4', verbose=False, logger=None)
        del seq
        gc.collect()


if __name__ == '__main__':
    args = parse_args()
    generate_boxes(hos_v3=args.input, sample_frames=args.sample_frames,
                   index=args.index)
