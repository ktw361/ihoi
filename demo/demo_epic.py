import argparse
import os
import os.path as osp
from demo import demo_utils

import numpy as np
from PIL import Image

# from renderer.screen_free_visualizer import Visualizer

from nnutils.hand_utils import ManopthWrapper
from nnutils.handmocap import get_handmocap_predictor, process_mocap_predictions, get_handmocap_detector
from nnutils.hoiapi import get_hoi_predictor, vis_hand_object

from datasets.epic_inf import EpicInference


def get_args():
    parser = argparse.ArgumentParser(
        description="Simple Epic inference")
    parser.add_argument(
        "--image_sets", 
        default='/home/skynet/Zhifan/data/epic_analysis/clean_frame_debug.txt')
    parser.add_argument("--out", default="output", help="Dir to save output.")
    parser.add_argument("--view", default="ego_centric", help="Dir to save output.")
    parser.add_argument("--merge_hand_mask", dest='merge_hand_mask', action='store_true')
    parser.add_argument("--no-merge_hand_mask", dest='merge_hand_mask', action='store_false')
    parser.set_defaults(merge_hand_mask=True)

    parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        default='weights/mow'
    )
    parser.add_argument("opts",  default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    return args


def main(args):

    # visualizer = Visualizer('pytorch3d')
    dataset = EpicInference(
        image_sets=args.image_sets,
        merge_hand_mask=args.merge_hand_mask)
    # bbox_detector = get_handmocap_detector(args.view)
    
    for idx, (image, hand_bbox_list, object_mask) in enumerate(dataset):
        # print(image.shape)

        # res_img = visualizer.visualize(image, hand_bbox_list = hand_bbox_list)
        # demo_utils.save_image(res_img, osp.join(args.out, 'hand_bbox.jpg'))

        hand_predictor = get_handmocap_predictor()
        mocap_predictions = hand_predictor.regress(
            image[..., ::-1], hand_bbox_list
        )
        # object_mask = np.ones_like(image[..., 0]) * 255

        # predict hand-held object
        hand_wrapper = ManopthWrapper().to('cpu')
        data = process_mocap_predictions(
            mocap_predictions, image, hand_wrapper, mask=object_mask
        )

        hoi_predictor = get_hoi_predictor(args)
        output = hoi_predictor.forward_to_mesh(data)
        vid, frame_idx = dataset.get_vid_frame(idx)
        vis_hand_object(output, data, image, 
                        osp.join(args.out, f"{vid}_{frame_idx}"))


if __name__ == "__main__":
    main(get_args())
