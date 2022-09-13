import argparse

import tqdm
import cv2
import os
import os.path as osp
import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets.epic_clip import EpicClipDataset
from homan.hand_forwarder import HandForwarder, init_hand_forwarder
from nnutils.handmocap import get_handmocap_predictor, collate_mocap_hand

from temporal.optim_plan import optimize_hand, smooth_hand_pose


from libzhifan import odlib
odlib.setup(order='xywh')

""" Run ihoi but with differentiable based pose optimizer. 

Note: don't delete this script as it can be used to Sanity-check FrankMocap

"""


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

    parser.add_argument("--show-bbox", action='store_true',
                        help='Show bounding boxes')
    parser.add_argument("--show-before", action='store_true',
                        help="Show original hand mesh before optimization.")

    args = parser.parse_args()
    return args


def main(args):
    dataset = EpicClipDataset(
        image_sets='/home/skynet/Zhifan/data/epic_analysis/gt_clips.json',
        sample_frames=20)
    hand_predictor = get_handmocap_predictor()
    if args.index >= 0:
        fit_scene(dataset, hand_predictor, args.index,
                  args.out, args.show_bbox, args.show_before)
        return
    
    for index in tqdm.trange(len(dataset)):
        try:
            fit_scene(dataset, hand_predictor, index,
                    args.out, args.show_bbox, args.show_before)
        except:
            continue


def fit_scene(dataset: EpicClipDataset, 
              hand_predictor,
              index: int, 
              out_dir: str,
              show_bbox=False,
              show_before=False):
    os.makedirs(out_dir, exist_ok=True)

    info = dataset.data_infos[index]
    images, hand_bbox_dicts, side, obj_bboxes, hand_masks, obj_masks, cat = dataset[index]
    print(f"Fit: {info}")

    overlay_mask = False
    if overlay_mask:
        for i in range(len(images)):
            masked_img = images[i].copy()
            masked_img[hand_masks[i] == 1, ...] = (255, 255, 255)
            img = cv2.addWeighted(images[i], 0.9, masked_img, 0.1, 1.0)
            images[i] = img

    """ Process all hands """
    mocap_predictions = []
    for img, hand_dict in zip(images, hand_bbox_dicts):
        mocap_pred = hand_predictor.regress(
            img[..., ::-1], [hand_dict]
        )
        mocap_predictions += mocap_pred
    one_hands = collate_mocap_hand(mocap_predictions, side)

    if show_bbox:
        fig = dataset.visualize_bboxes(index)
        save_name = f"{info.vid}_{info.gt_frame}_bbox.png"
        save_name = osp.join(out_dir, save_name)
        fig.savefig(save_name)

    hmod = init_hand_forwarder(one_hands, images, side, obj_bboxes, hand_masks)
    # hmod.checkpoint()

    if show_before:
        fig = hmod.render_grid()
        save_name = f"{info.vid}_{info.gt_frame}_before.png"
        save_name = osp.join(out_dir, save_name)
        fig.savefig(save_name)

    hmod = smooth_hand_pose(hmod)
    hmod = optimize_hand(hmod)

    fig = hmod.render_grid()
    save_name = f"{info.vid}_{info.gt_frame}.png"
    save_name = osp.join(out_dir, save_name)
    fig.savefig(save_name)

    fig = visualize_loss(hmod.loss_records)
    save_name = f"{info.vid}_{info.gt_frame}_loss.png"
    save_name = osp.join(out_dir, save_name)
    fig.savefig(save_name)


def visualize_loss(loss_records: dict):
    fig = plt.figure()
    for k, v in loss_records.items():
        plt.plot(v)

    plt.legend(loss_records.keys())
    plt.show()
    return fig


if __name__ == "__main__":
    main(get_args())