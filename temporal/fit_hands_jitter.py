import argparse

import cv2
import os.path as osp
import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets.epic_clip import EpicClipDataset
from homan.hand_forwarder import HandForwarder, init_hand_forwarder
from nnutils.handmocap import get_handmocap_predictor, collate_mocap_hand

from temporal.optim_plan import optimize_hand
from temporal.utils import jitter_box, median_vector_index


from libzhifan import odlib
odlib.setup(order='xywh')

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

    parser.add_argument("--show-bbox", action='store_true',
                        help='Show bounding boxes')
    parser.add_argument("--show-before", action='store_true',
                        help="Show original hand mesh before optimization.")

    args = parser.parse_args()
    return args


def main(args):
    os.makedirs(args.out, exist_ok=True)

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
        boxes = jitter_box(hand_dict[side], ratio=0.005, num=5)
        hand_dict_list = []
        for box in boxes:
            hand_dict_list.append({side: box})
            
        mocap_pred_list = hand_predictor.regress(
            img[..., ::-1], hand_dict_list
        )
        poses = np.concatenate(
            [v[side]['pred_hand_pose'] for v in mocap_pred_list], axis=0)
        ind = median_vector_index(torch.as_tensor(poses))
        mocap_predictions.append(mocap_pred_list[ind])
    one_hands = collate_mocap_hand(mocap_predictions, side)

    if args.show_bbox:
        fig = visualize_bboxes(
            images, hand_bbox_dicts, side, obj_bboxes, hand_masks, obj_masks)
        save_name = f"{info.vid}_{info.gt_frame}_bbox.png"
        save_name = osp.join(args.out, save_name)
        fig.savefig(save_name)

    hmod = init_hand_forwarder(one_hands, images, side, obj_bboxes, hand_masks)
    # hmod.checkpoint()

    if args.show_before:
        fig = hmod.render_grid()
        save_name = f"{info.vid}_{info.gt_frame}_before.png"
        save_name = osp.join(args.out, save_name)
        fig.savefig(save_name)

    hmod = optimize_hand(hmod)

    fig = hmod.render_grid()
    save_name = f"{info.vid}_{info.gt_frame}.png"
    save_name = osp.join(args.out, save_name)
    fig.savefig(save_name)


def visualize_bboxes(images,
                     hand_bbox_dicts,
                     side,
                     obj_bboxes,
                     hand_masks,
                     obj_masks):
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


if __name__ == "__main__":
    main(get_args())