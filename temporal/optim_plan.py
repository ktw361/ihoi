from typing import NamedTuple, Union
import tqdm
import numpy as np
import torch
from homan.ho_forwarder_v2 import HOForwarderV2Vis, HOForwarderV2Impl
from temporal.utils import choose_with_softmax


""" Different HO optimization plans. """


def smooth_hand_pose(homan: HOForwarderV2Impl,
                     lr=1e-2,
                     thresh=1e-3,
                     verbose=True):
    """ smooth pca pose """
    optimizer = torch.optim.Adam([
        {
            'params': [
                homan.mano_pca_pose
            ],
            'lr': lr
        }
    ])

    max_steps = 500
    for _ in range(max_steps):
        optimizer.zero_grad()
        pca_loss = homan.loss_pca_interpolation().sum()
        if pca_loss < thresh:
            break
        pca_loss.backward()
        optimizer.step()

    return homan


def optimize_hand(homan: HOForwarderV2Impl,
                  lr=1e-2,
                  num_steps=100,
                  verbose=True) -> HOForwarderV2Impl:
    optimizer = torch.optim.Adam([
        {
            'params': [
                homan.rotations_hand,
                homan.translations_hand,
            ],
            'lr': lr
        }
    ])

    loss_records = {
        'total': [],
        'sil': [],
        'pca': [],
        'rot': [],
        'transl': [],
    }
    for step in range(num_steps):
        optimizer.zero_grad()
        tot_loss, loss_dict = homan.forward_hand()
        if verbose and step % 10 == 0:
            print(f"Step {step}, tot = {tot_loss.item():.04f}, ", end=' ')
            for k, v in loss_dict.items():
                print(f"{k} = {v.item():.04f}", end=' ')
            print()
        loss_records['total'].append(tot_loss.item())
        for k, v in loss_dict.items():
            loss_records[k].append(v.item())

        tot_loss.backward()
        optimizer.step()
    
    homan.loss_records = loss_records

    return homan


class ObjectParams(NamedTuple):
    translations_object: torch.Tensor
    rotations_object: torch.Tensor
    scale_object: torch.Tensor


def exhaust_optim_obj_pose(homan: HOForwarderV2Impl,
                           frame_idx: int,
                           lr=1e-3,
                           max_iters=5000,
                           stop_thresh=1e-6):
    """
    Args:
        stop_thresh: Stop if difference between two loss less than this number.

    """
    optimizer = torch.optim.Adam([
        {
            'params': [
                homan.rotations_object,
                homan.translations_object,
                homan.scale_object,
            ],
            'lr': lr
        }
    ])

    best_loss_single = torch.tensor(np.inf)

    loss_prev = torch.tensor(np.inf)
    with tqdm.tqdm(total=max_iters) as loop:
        for _ in range(max_iters):
            optimizer.zero_grad()
            l_obj_dict = homan.simple_obj_sil(frame_idx=frame_idx)
            l_obj_mask = l_obj_dict['mask']  # (B,)
            l_contact = homan.loss_chamfer(batch_reduction='sum')  # homan.loss_contact()
            l_depth = homan.loss_ordinal_depth()

            loss = l_obj_mask.sum() + l_contact.sum() + l_depth.sum()
            diff = torch.abs(loss_prev - loss)
            if diff < stop_thresh:
                break
            loss_prev = loss
            loss.backward()
            optimizer.step()
            if loss.min() < best_loss_single:
                best_rots_single = homan.rotations_object.detach().clone()
                best_trans_single = homan.translations_object.detach().clone()
                best_scale_single = homan.scale_object.detach().clone()
                best_loss_single = loss.min()
            loop.set_description(f"obj loss: {best_loss_single.item():.3g}, diff: {diff:.4ikjkg}")
            loop.update()

    params = ObjectParams(best_trans_single, best_rots_single, best_scale_single)
    return params


def find_optimal_obj_pose(homan: HOForwarderV2Impl,
                          frame_idx: int,
                          num_iterations=50,
                          lr=1e-3,
                          max_iters=5000,
                          stop_thresh=1e-3,
                          sort_best=True):
    """
    Args:
        stop_thresh: Stop if difference between two loss less than this number.

    """
    raise ValueError("Obsolete")
    optimizer = torch.optim.Adam([
        {
            'params': [
                homan.rotations_object,
                homan.translations_object,
                homan.scale_object,
            ],
            'lr': lr
        }
    ])

    best_losses = torch.tensor(np.inf)
    best_rots = None
    best_trans = None
    best_loss_single = torch.tensor(np.inf)

    with tqdm.tqdm(total=num_iterations) as loop:
        for _ in range(num_iterations):
            optimizer.zero_grad()
            l_obj_dict = homan.simple_obj_sil(frame_idx=frame_idx)
            l_obj_mask = l_obj_dict['mask']  # (B,)

            losses = l_obj_mask
            loss = losses[frame_idx].sum()
            loss.backward()
            optimizer.step()
            if losses.min() < best_loss_single:
                ind = torch.argmin(losses)
                best_loss_single = losses[ind]
                best_rots_single = homan.rotations_object[ind].detach().clone()
                best_trans_single = homan.translations_object[ind].detach().clone()
            loop.set_description(f"obj loss: {best_loss_single.item():.3g}")
            loop.update()

    best_rots = homan.rotations_object
    best_trans = homan.translations_object
    best_losses = losses
    if sort_best:
        inds = torch.argsort(best_losses)
        num_obj_init = homan.num_obj_init
        best_losses = best_losses[inds][:num_obj_init].detach().clone()
        best_trans = best_trans[inds][:num_obj_init].detach().clone()
        best_rots = best_rots[inds][:num_obj_init].detach().clone()
    # Add best ever:

    if sort_best:
        best_rots = torch.cat((best_rots_single.unsqueeze(0), best_rots[:-1]),
                              0)
        best_trans = torch.cat(
            (best_trans_single.unsqueeze(0), best_trans[:-1]), 0)
    homan.rotations_object = torch.nn.Parameter(best_rots)
    homan.translations_object = torch.nn.Parameter(best_trans)
    return homan


def optimize_hand_allmask(homan: HOForwarderV2Vis,
                          lr=1e-2,
                          num_steps=100,
                          vis_interval=-1,
                          vis=False,
                          writer=None):
    if hasattr(homan, 'info'):
        info = homan.info
        prefix = f'{info.vid}_{info.gt_frame}'

    optimizer = torch.optim.Adam([
        {
            'params': [
                # homan.rotations_hand,
                # homan.translations_hand, # (B)
                homan.rotations_object,  # (1,)
                homan.translations_object,
                homan.scale_object,
            ],
            'lr': lr
        }
    ])

    with tqdm.tqdm(total=num_steps) as loop:
        for step in range(num_steps):
            optimizer.zero_grad()
            l_pca = homan.loss_pca_interpolation().sum()
            l_hand_pose = homan.loss_hand_rot().sum() + homan.loss_hand_transl().sum()
            l_sil_hand = homan.loss_sil_hand(compute_iou=False, func='l2_iou')  # (B,)

            l_obj_dict = homan.forward_obj_pose_render(loss_only=True, func='l2_iou')  # (B,N)
            l_obj_mask = l_obj_dict['mask']
            l_obj_offscreen = l_obj_dict['offscreen']
            # l_obj_center = homan.diff_proj_center()

            l_contact = homan.loss_chamfer(batch_reduction='sum')  # homan.loss_contact()
            # l_contact = homan.loss_nearest_dist().sum()
            # l_collision = homan.loss_collision()
            l_depth = homan.loss_ordinal_depth()

            tot_loss = 0.1 * l_sil_hand.sum() + \
                0.1 * (l_obj_mask.sum() + l_obj_offscreen.sum()) +\
                l_pca + l_hand_pose +\
                l_depth + 10*l_contact #+ l_collision

            if vis_interval > 0 and step % vis_interval == 0:
                print(
                    # f"min_d:{l_min_d.item():.3f}"
                    f"pca:{l_pca.item():.3f} "
                    f"sil_hand:{l_sil_hand.sum().item():.3f} "
                    f"obj_mask:{l_obj_mask.sum().item():.3f} "
                    # f"obj_center:{l_obj_center.sum().item():.3f} "
                    f"contact:{l_contact.sum().item():.3f} "
                    f"hand_pose:{l_hand_pose.item():.3f}")
                if writer is None:
                    if vis:
                        _ = homan.render_grid(obj_idx=0, low_reso=True)
                else:
                    img = homan.render_grid_np(obj_idx=0)
                    writer.add_image(tag=f'{prefix}', 
                                     img_tensor=img.transpose(2, 0, 1),
                                     global_step=step)

            tot_loss.backward()
            optimizer.step()
            loop.set_description(f"tot_loss: {tot_loss.item():.3g}")
            loop.update()

    return homan


def sampled_obj_optimize(homan: HOForwarderV2Vis,
                         lr=1e-2,
                         num_epochs=50,
                         num_iters=2000,
                         temperature=100.,
                         ratio=0.5,
                         weights=None,
                         vis_interval=-1,
                         vis=False,
                         writer=None):
                         
    if hasattr(homan, 'info'):
        info = homan.info
        prefix = f'{info.vid}_{info.gt_frame}'

    optimizer = torch.optim.Adam([
        {
            'params': [
                homan.rotations_object,  # (1,)
                homan.translations_object,
                homan.scale_object,
            ],
            'lr': lr
        }
    ])

    weights = homan.rotations_hand.new_zeros([homan.bsize]) if weights is None else weights

    for e in range(num_epochs):

        sample_indices = choose_with_softmax(
            weights, temperature=temperature, ratio=ratio)
        print(f"Sample {sample_indices} at epoch {e}, weights = {weights.tolist()}")

        with tqdm.tqdm(total=num_iters) as loop:
            for step in range(num_iters):
                optimizer.zero_grad()

                v_hand = homan.get_verts_hand()[sample_indices, ...]
                v_obj = homan.get_verts_object()[sample_indices, ...]

                l_obj_dict = homan.forward_obj_pose_render(
                    sample_indices=sample_indices)  # (B,N)
                l_obj_mask = l_obj_dict['mask']
                l_obj_offscreen = l_obj_dict['offscreen']
                l_contact = homan.loss_chamfer(
                    v_hand=v_hand, v_obj=v_obj, sample_indices=sample_indices)
                # l_depth = homan.loss_ordinal_depth(v_hand=v_hand, v_obj=v_obj)
                min_dist = homan.loss_nearest_dist(v_hand=v_hand, v_obj=v_obj).min()

                # Accumulate
                l_obj_mask = l_obj_mask.sum()
                l_obj_offscreen = torch.zeros_like((0.1 * l_obj_offscreen.sum()))
                # l_depth = torch.zeros_like(l_depth)
                l_contact = torch.zeros_like(100 * l_contact)
                tot_loss = l_obj_mask + l_obj_offscreen + l_contact

                if vis_interval > 0 and step % vis_interval == 0:
                    print(
                        f"obj_mask:{l_obj_mask.item():.3f} "
                        f"obj_offscrn:{l_obj_offscreen.item():.3f} "
                        # f"depth:{l_depth.item():.3f} "
                        f"contact:{l_contact.item():.3f} "
                        f"min_dist: {min_dist:.3f} "
                        )
                    if writer is None:
                        if vis:
                            _ = homan.render_grid(obj_idx=0, low_reso=True)
                    else:
                        img = homan.render_grid_np(obj_idx=0)
                        writer.add_image(tag=f'{prefix}', 
                                        img_tensor=img.transpose(2, 0, 1),
                                        global_step=step)

                tot_loss.backward()
                optimizer.step()
                loop.set_description(f"tot_loss: {tot_loss.item():.3g}")
                loop.update()
    
        # Update weights
        weights[sample_indices] -= tot_loss

    return homan, weights