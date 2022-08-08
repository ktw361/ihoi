import argparse

import numpy as np
import torch
from datasets.epic_clip import EpicClipDataset
from homan.ho_forwarder import HOForwarder
from nnutils.hand_utils import ManopthWrapper
from nnutils.handmocap import get_handmocap_predictor, collate_mocap_hand
from nnutils import image_utils

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
        image_sets=args.image_sets)

    device = 'cuda'
    hand_wrapper_left = ManopthWrapper(flat_hand_mean=False, side='left').to(device)
    hand_wrapper_right = ManopthWrapper(flat_hand_mean=False, side='right').to(device)
    obj_loader = OBJLoader()
    hand_predictor = get_handmocap_predictor()

    index = args.index
    info = dataset.data_infos[index]
    images, hand_bbox_dicts, side, obj_bboxes, hand_masks, obj_masks, cat = dataset[index]
    if side == 'left_hand':
        hand_wrapper_flat = hand_wrapper_left
    elif side == 'right_hand':
        hand_wrapper_flat = hand_wrapper_right
    else:
        raise ValueError

    """ Process all hands """
    mocap_predictions = []
    for img, hand_dict in zip(images, hand_bbox_dicts):
        mocap_pred = hand_predictor.regress(
            img[..., ::-1], [hand_dict]
        )
        mocap_predictions += mocap_pred
    one_hands = collate_mocap_hand(mocap_predictions, side)

    """ Process key-frame """
    gt_frame, start, end = info.gt_frame, info.start, info.end
    assert gt_frame >= start
    # t = gt_frame - start

    pose_machine = PoseOptimizer(
        one_hands, obj_loader, hand_wrapper_flat,
        device=device,
    )
    pose_machine.fit_obj_pose(
        images, obj_bboxes, obj_masks, cat,
        put_hand_transform=True,
        num_initializations=10, num_iterations=5,
        sort_best=False, debug=False, viz=False
    )
    # Optimize K cluster scales
    # K = 10
    # obj_pose_results = pose_machine.pose_model.clustered_results(K=K)

    torch.cuda.empty_cache()
    homan = init_ho_forwarder(
        pose_machine,
        obj_bboxes,
        hand_masks, obj_masks,
        side,
        pose_idx=0
    )

    homan = optimize_scale(homan, num_steps=100, lr=1e-2, verbose=True)

    return
    """ Now we have K sequence of {gt, ..., end}... """
    # sample_dir = osp.join(args.out, f"{info.vid}_{info.start}")


def init_ho_forwarder(pose_machine,
                      obj_bbox,
                      mask_hand,
                      mask_obj,
                      hand_side,
                      pose_idx,
                      return_args=False) -> HOForwarder:
    # Optimize K cluster scales
    K = 10
    obj_pose_results = pose_machine.pose_model.clustered_results(K=K)
    # obj_pose_results = pose_machine.pose_model.fitted_results

    bsize = len(pose_machine)
    obj_bbox_squared = image_utils.square_bbox_xywh(
        obj_bbox, pose_machine.ihoi_box_expand).int()
    obj_mask_patch = pose_machine.batch_pad_and_crop(
        mask_obj, obj_bbox_squared, pose_machine.rend_size)
    hand_mask_patch = pose_machine.batch_pad_and_crop(
        mask_hand, obj_bbox_squared, pose_machine.rend_size)
    mano_pca_pose = pose_machine.recover_pca_pose()
    mano_rot = torch.zeros([bsize, 3], device=mano_pca_pose.device)
    mano_trans = torch.zeros([bsize, 3], device=mano_pca_pose.device)
    camintr = pose_machine.pose_model.camera_K  # could be pose_machine.ihoi_cam

    homan_kwargs = dict(
        translations_object = obj_pose_results.translations[[pose_idx]],
        rotations_object = obj_pose_results.rotations[[pose_idx]],
        verts_object_og = pose_machine.pose_model.vertices,
        faces_object = pose_machine.pose_model.faces[None],
        translations_hand = pose_machine.hand_translation,
        rotations_hand = pose_machine.hand_rotation,
        verts_hand_og = pose_machine.hand_verts,
        hand_sides = [hand_side],
        mano_trans = mano_trans,
        mano_rot = mano_rot,
        mano_betas = pose_machine.pred_hand_betas,
        mano_pca_pose = pose_machine.recover_pca_pose(),
        faces_hand = pose_machine.hand_faces,

        scale_object = 1.0,
        scale_hand = 1.0,

        camintr = camintr,
        target_masks_hand = hand_mask_patch,
        target_masks_object = obj_mask_patch,

        image_size = pose_machine.rend_size,
        ihoi_img_patch=pose_machine._image_patch
        )

    for k, v in homan_kwargs.items():
        if hasattr(v, 'device'):
            homan_kwargs[k] = v.to('cuda')
    if return_args:
        return homan_kwargs
    homan = HOForwarder(**homan_kwargs).cuda()
    # torch.cuda.empty_cache()
    return homan


if __name__ == "__main__":
    main(get_args())
