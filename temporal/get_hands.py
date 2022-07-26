import argparse
import os
import os.path as osp

import torch
from nnutils.hand_utils import ManopthWrapper
from nnutils.handmocap import get_handmocap_predictor
from datasets.epic_clip import EpicClipDataset, CachedEpicClipDataset

from obj_pose.pose_optimizer import PoseOptimizer, SavedContext
from obj_pose.obj_loader import OBJLoader


""" Run ihoi but with differentiable based pose optimizer. """


def get_args():
    parser = argparse.ArgumentParser(
        description="Simple Epic inference")
    parser.add_argument(
        "index", type=int,
        help="Index into gt_clips.json")
    parser.add_argument(
        "--image_sets",
        default='/home/skynet/Zhifan/data/epic_analysis/gt_clips.json')
    parser.add_argument("--out", default="output/temporal", help="Dir to save output.")

    parser.add_argument("opts",  default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    return args


def main(args):

    dataset = CachedEpicClipDataset(
        image_sets=args.image_sets)

    device = 'cuda'
    hand_wrapper_left = ManopthWrapper(flat_hand_mean=False, side='left').to(device)
    hand_wrapper_right = ManopthWrapper(flat_hand_mean=False, side='right').to(device)
    # obj_loader = OBJLoader()
    hand_predictor = get_handmocap_predictor()

    index = args.index
    info = dataset.data_infos[index]
    images, hand_bbox_dicts, side, obj_bboxes, hand_masks, obj_masks, cat = dataset[index]

    one_hands = []
    for frame_idx in range(info.start, info.end+1):
        i = frame_idx - info.start
        mocap_predictions = hand_predictor.regress(
            images[i, ..., ::-1], [hand_bbox_dicts[i]]
        )

        if side == 'left_hand':
            hand_wrapper_flat = hand_wrapper_left
        elif side == 'right_hand':
            hand_wrapper_flat = hand_wrapper_right
        else:
            raise ValueError
        one_hand = mocap_predictions[0][side]
        # predict object
        # pose_machine = PoseOptimizer(
        #     one_hand, obj_loader, hand_wrapper_flat,
        #     device=device,
        # )
        # pose_machine.fit_obj_pose(
        #     image, obj_bbox, mask_obj, cat,
        #     num_initializations=400, num_iterations=50,
        #     sort_best=False, debug=False, viz=False
        # )
        one_hands.append(one_hand)

    sample_dir = osp.join(args.out, f"{info.vid}_{info.start}")
    # context = SavedContext(
    #     pose_machine=pose_machine,
    #     obj_bbox=obj_bbox,
    #     mask_hand=mask_hand,
    #     mask_obj=mask_obj,
    #     hand_side=side.replace('_hand', '')
    # )
    os.makedirs(sample_dir, exist_ok=True)
    torch.save(one_hands, osp.join(sample_dir, 'one_hands.pt'))


if __name__ == "__main__":
    main(get_args())
