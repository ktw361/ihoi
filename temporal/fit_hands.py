import argparse

import os.path as osp
import numpy as np
import torch
from datasets.epic_clip import EpicClipDataset
from homan.hand_forwarder import HandForwarder, init_hand_forwarder
from nnutils.hand_utils import ManopthWrapper
from nnutils.handmocap import get_handmocap_predictor, collate_mocap_hand
from nnutils import image_utils

from temporal.optim_plan import optimize_hand

from obj_pose.pose_optimizer import PoseOptimizer, SavedContext
from obj_pose.obj_loader import OBJLoader


""" Run ihoi but with differentiable based pose optimizer. """


def get_args():
    parser = argparse.ArgumentParser(
        description="Simple Epic inference")
    parser.add_argument(
        "index", type=int, default=5,
        help="Index into gt_clips.json")
    parser.add_argument(
        "--image_sets",
        default='/home/skynet/Zhifan/data/epic_analysis/gt_clips.json')
    parser.add_argument("--out", default="output/temporal", help="Dir to save output.")

    parser.add_argument("opts",  default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    return args


def main(args):

    dataset = EpicClipDataset(
    image_sets='/home/skynet/Zhifan/data/epic_analysis/gt_clips.json',
    sample_frames=20)

    device = 'cuda'
    hand_predictor = get_handmocap_predictor()

    index = args.index
    info = dataset.data_infos[index]
    images, hand_bbox_dicts, side, obj_bboxes, hand_masks, obj_masks, cat = dataset[index]

    """ Process all hands """
    mocap_predictions = []
    for img, hand_dict in zip(images, hand_bbox_dicts):
        mocap_pred = hand_predictor.regress(
            img[..., ::-1], [hand_dict]
        )
        mocap_predictions += mocap_pred
    one_hands = collate_mocap_hand(mocap_predictions, side)

    hmod = init_hand_forwarder(one_hands, images, side, obj_bboxes, hand_masks)
    # hmod.checkpoint()

    hmod = optimize_hand(hmod)
    fig = hmod.render_grid()
    save_name = f"{info.vid}_{info.gt_frame}.png"
    save_name = osp.join(args.out, save_name)
    fig.savefig(save_name)


if __name__ == "__main__":
    main(get_args())
