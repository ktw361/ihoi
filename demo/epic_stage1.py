import argparse
import os
import os.path as osp

import torch
import trimesh
from nnutils.hand_utils import ManopthWrapper
from nnutils.handmocap import get_handmocap_predictor
from datasets.epic_inf import EpicInference

from obj_pose.pose_optimizer import PoseOptimizer, SavedContext
from obj_pose.obj_loader import OBJLoader


""" Run ihoi but with differentiable based pose optimizer. """


def get_args():
    parser = argparse.ArgumentParser(
        description="Simple Epic inference")
    parser.add_argument(
        "--image_sets",
        default='/home/skynet/Zhifan/data/epic_analysis/clean_frame_debug.txt')
    parser.add_argument("--out", default="output/stage1", help="Dir to save output.")
    parser.add_argument("--view", default="ego_centric", help="Dir to save output.")

    parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        default='weights/ho3d'
    )
    parser.add_argument("opts",  default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    return args


def main(args):

    dataset = EpicInference(
        image_sets=args.image_sets)

    device = 'cuda'
    # predict hand
    hand_wrapper_left = ManopthWrapper(flat_hand_mean=False, side='left').to(device)
    hand_wrapper_right = ManopthWrapper(flat_hand_mean=False, side='right').to(device)
    obj_loader = OBJLoader()

    for idx, (image, hand_bbox_dict, side, obj_bbox,
              mask_hand, mask_obj, cat) in enumerate(dataset):

        if idx % 10 == 0:
            print(f"Processing {idx}/{len(dataset)}")

        hand_predictor = get_handmocap_predictor()
        mocap_predictions = hand_predictor.regress(
            image[..., ::-1], [hand_bbox_dict]
        )

        if side == 'left_hand':
            hand_wrapper_flat = hand_wrapper_left
        elif side == 'right_hand':
            hand_wrapper_flat = hand_wrapper_right
        else:
            raise ValueError
        one_hand = mocap_predictions[0][side]
        # predict object
        pose_machine = PoseOptimizer(
            one_hand, obj_loader, hand_wrapper_flat,
            device=device,
        )
        pose_machine.fit_obj_pose(
            image, obj_bbox, mask_obj, cat,
            num_initializations=400, num_iterations=50,
            sort_best=False, debug=False, viz=False
        )
        vid, frame_idx = dataset.get_vid_frame(idx)
        sample_dir = osp.join(args.out, f"{vid}_{frame_idx}")
        context = SavedContext(
            pose_machine=pose_machine,
            obj_bbox=obj_bbox,
            mask_hand=mask_hand,
            mask_obj=mask_obj,
            hand_side=side.replace('_hand', '')
        )
        os.makedirs(sample_dir, exist_ok=True)
        torch.save(context, osp.join(sample_dir, 'saved_context.pth'))


if __name__ == "__main__":
    main(get_args())
