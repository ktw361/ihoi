from typing import NamedTuple
import argparse
import os
import os.path as osp

import numpy as np
import torch
from datasets.epic_clip import EpicClipDataset, CachedEpicClipDataset
from homan.ho_forwarder import HOForwarder
from nnutils.hand_utils import ManopthWrapper
from nnutils.handmocap import get_handmocap_predictor
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


def init_ho_forwarder(pose_machine, 
                      obj_bbox, 
                      mask_hand, 
                      mask_obj, 
                      obj_pose_results,
                      hand_side,
                      pose_idx) -> HOForwarder:
    obj_bbox_squared = image_utils.square_bbox_xywh(
        obj_bbox, pose_machine.ihoi_box_expand).int()
    obj_mask_patch = pose_machine.pad_and_crop(
        mask_obj, obj_bbox_squared, pose_machine.rend_size)
    hand_mask_patch = pose_machine.pad_and_crop(
        mask_hand, obj_bbox_squared, pose_machine.rend_size)
    mano_pca_pose = pose_machine.recover_pca_pose()
    mano_rot = torch.zeros([1, 3], device=mano_pca_pose.device)
    mano_trans = torch.zeros([1, 3], device=mano_pca_pose.device)
    camintr = pose_machine.pose_model.K  # could be pose_machine.ihoi_cam

    homan_kwargs = dict(
        translations_object = obj_pose_results.translations[[pose_idx]],
        rotations_object = obj_pose_results.rotations[[pose_idx]],
        verts_object_og = pose_machine.pose_model.vertices,
        faces_object = pose_machine.pose_model.faces[[pose_idx]],
        translations_hand = pose_machine.hand_translation,
        rotations_hand = pose_machine.hand_rotation,
        verts_hand_og = pose_machine.hand_verts,
        hand_sides = [hand_side],
        mano_trans = mano_trans,
        mano_rot = mano_rot,
        mano_betas = pose_machine.pred_hand_betas,
        mano_pca_pose = pose_machine.recover_pca_pose(),
        faces_hand = pose_machine.hand_faces,
        
        scale_object = 1.5,
        scale_hand = 1.0,

        camintr = camintr,
        target_masks_hand = obj_mask_patch,
        target_masks_object = hand_mask_patch,

        image_size = pose_machine.rend_size,
        ihoi_img_patch=pose_machine._image_patch
        )

    for k, v in homan_kwargs.items():
        if hasattr(v, 'device'):
            homan_kwargs[k] = v.to('cuda')
    homan = HOForwarder(**homan_kwargs).cuda()
    # torch.cuda.empty_cache()
    return homan


def optimize_scale(homan, 
                   num_steps=100, 
                   lr=1e-2,
                   verbose=False) -> HOForwarder:
    scale_weights = dict(
        lw_pca=0.0,
        lw_collision=1.0,
        lw_contact=1.0,
        lw_sil_obj=1.0,
        lw_sil_hand=0.0,
        lw_inter=1.0,

        lw_scale_obj=0.0,  # mean deviation loss
        lw_scale_hand=0.0,
        lw_depth=1.0
    )

    optimizer = torch.optim.Adam([
        {
            'params': [homan.scale_object],
            'lr': lr
        }
    ])

    for step in range(num_steps):
        optimizer.zero_grad()
        loss_dict, metric_dict = homan(loss_weights=scale_weights)
        loss_dict_weighted = {
            k: loss_dict[k] * scale_weights[k.replace("loss", "lw")]
            for k in loss_dict
        }
        loss = sum(loss_dict_weighted.values())
        if verbose and step % 10 == 0:
            print(f"Step {step}, total loss = {loss.item():.05f}: ", end='')
            for k, v in loss_dict_weighted.items():
                print(f"{k.replace('loss_', '')}, {v.item():.05f}", end=', ')
            print('\n', end='')
            print(f"scale={homan.scale_object.item()}")

        loss.backward()
        optimizer.step()

    return homan


def reinit_ho_forwarder(pose_machine,
                        obj_bbox,
                        mask_hand,
                        mask_obj,
                        hand_side,
                        homan_prev) -> HOForwarder:
    _, target_masks_object, target_masks_hand = pose_machine._get_bbox_and_crop(
        mask_obj, mask_hand, obj_bbox)  # from global to local
    mano_pca_pose = pose_machine.recover_pca_pose()
    mano_rot = torch.zeros([1, 3], device=mano_pca_pose.device)
    mano_trans = torch.zeros([1, 3], device=mano_pca_pose.device)
    camintr = pose_machine.ihoi_cam.get_K() / pose_machine.rend_size  # could be pose_machine.ihoi_cam

    homan_kwargs = dict(
        translations_object = homan_prev.translations_object.detach().clone(),
        rotations_object = homan_prev.rotations_object.detach().clone(),
        verts_object_og = homan_prev.verts_object_og,
        faces_object = homan_prev.faces_object,
        translations_hand = pose_machine.hand_translation,
        rotations_hand = pose_machine.hand_rotation,
        verts_hand_og = pose_machine.hand_verts,
        hand_sides = [hand_side],
        mano_trans = mano_trans,
        mano_rot = mano_rot,
        mano_betas = pose_machine.pred_hand_betas,
        mano_pca_pose = pose_machine.recover_pca_pose(),
        faces_hand = pose_machine.hand_faces,
        
        scale_object = homan_prev.scale_object.detach().clone(),
        scale_hand = homan_prev.scale_hand.detach().clone(),

        camintr = camintr,
        target_masks_hand = torch.as_tensor(target_masks_hand),
        target_masks_object = torch.as_tensor(target_masks_object),

        image_size = pose_machine.rend_size,
        ihoi_img_patch=pose_machine._image_patch
        )

    for k, v in homan_kwargs.items():
        if hasattr(v, 'device'):
            homan_kwargs[k] = v.to('cuda')
    homan = HOForwarder(**homan_kwargs).cuda()
    # torch.cuda.empty_cache()
    return homan


def optimize_temporal(homan, 
                      homan_prev=None,
                      num_iterations=100,
                      lr=1e-3,
                      ) -> HOForwarder:
    loss_weights = dict(
        lw_pca=0.0,
        lw_collision=1.0,
        lw_contact=1.0,
        lw_sil_obj=1.0,  # Is sil_obj harmful?
        lw_sil_hand=1.0,
        lw_inter=1.0,

        lw_scale_obj=0.0,  # mean deviation loss
        lw_scale_hand=0.0,
        lw_depth=0.01,
    )

    all_params = [
        homan.rotations_hand, 
        homan.translations_hand, 
        homan.mano_pca_pose,
        homan.translations_object,
        homan.rotations_object,
    ]
    optimizer = torch.optim.Adam([
        {
            'params': all_params,
            'lr': lr
        }   
    ])
    # homan(loss_weights, scale_object=scale_object2)
    for step in range(num_iterations):
        optimizer.zero_grad()
        loss_dict, metric_dict = homan(loss_weights=loss_weights)
        loss_dict_weighted = {
            k: loss_dict[k] * loss_weights[k.replace("loss", "lw")]
            for k in loss_dict
        }
        loss = sum(loss_dict_weighted.values())
        if step % 10 == 0:
            print(f"Step {step}, total loss = {loss.item():.05f}: ", end='')
            for k, v in loss_dict_weighted.items():
                print(f"{k.replace('loss_', '')}, {v.item():.05f}", end=', ')
            print('\n', end='')

        loss.backward()
        optimizer.step()
    return homan


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

    """ Process key-frame """
    gt_frame, start, end = info.gt_frame, info.start, info.end
    assert gt_frame >= start
    # t = gt_frame - start
    t = 0
    mocap_predictions = hand_predictor.regress(
        images[t, ..., ::-1], [hand_bbox_dicts[t]]
    )

    one_hand = mocap_predictions[0][side]
    pose_machine = PoseOptimizer(
        one_hand, obj_loader, hand_wrapper_flat,
        device=device,
    )
    pose_machine.fit_obj_pose(
        images[t], obj_bboxes[t], obj_masks[t], cat,
        num_initializations=400, num_iterations=50,
        sort_best=False, debug=False, viz=False
    )
    # Optimize K cluster scales
    K = 10
    obj_pose_results = pose_machine.pose_model.clustered_results(K=K)

    T = end - start + 1
    homan = np.empty((T, K), dtype=object)  # Lattice
    for k in range(K):
        homan[t, k] = init_ho_forwarder(
            pose_machine,
            obj_bboxes[t],
            hand_masks[t], obj_masks[t], 
            obj_pose_results,
            side,
            k)
        homan[t, k] = optimize_scale(homan[t, k])

    for frame_idx in range(gt_frame+1, end+1):
        t = frame_idx - info.start
        mocap_predictions = hand_predictor.regress(
            images[t, ..., ::-1], [hand_bbox_dicts[t]]
        )
        one_hand = mocap_predictions[0][side]
        for k in range(K):
            pose_machine = PoseOptimizer(
                one_hand, obj_loader, hand_wrapper_flat,
                device=device,
            )
            _ = pose_machine.finalize_without_fit(
                images[t], obj_bboxes[t], obj_masks[t])
            homan[t, k] = reinit_ho_forwarder(
                pose_machine, obj_bboxes[t],
                hand_masks[t], obj_masks[t], side,
                homan[t-1, k])
            homan[t, k] = optimize_temporal(homan[t, k], homan[t-1, k])
    
    """ Now we have K sequence of {gt, ..., end}... """
    # sample_dir = osp.join(args.out, f"{info.vid}_{info.start}")


if __name__ == "__main__":
    main(get_args())
