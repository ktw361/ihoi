from typing import Union
import tqdm
import numpy as np
import torch
from homan.ho_forwarder_v2 import HOForwarderV2Vis, HOForwarderV2Impl


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
                  verbose=True):
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


def find_optimal_obj_pose(homan: HOForwarderV2Impl,
                          cam_idx: int,
                          num_iterations=50,
                          lr=1e-3,
                          sort_best=True):
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
    best_trans_single = None

    with tqdm.tqdm(total=num_iterations) as loop:
        for _ in range(num_iterations):
            optimizer.zero_grad()
            l_obj_dict = homan.forward_obj_sil_frame(cam_idx=cam_idx)
            l_obj_mask = l_obj_dict['mask']  # (B,)

            losses = l_obj_mask
            loss = losses[cam_idx].sum()
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
                          writer=None):
    info = homan.info
    prefix = f'{info.vid}_{info.gt_frame}'

    optimizer = torch.optim.Adam([
        {
            'params': [
                homan.rotations_hand,
                homan.translations_hand, # (B)
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
            # l_min_d = homan.loss_nearest_dist().sum()
            l_pca = homan.loss_pca_interpolation().sum()
            l_hand_pose = homan.loss_hand_rot().sum() + homan.loss_hand_transl().sum()
            l_sil_hand = homan.loss_sil_hand(compute_iou=False, func='iou')  # (B,)
            l_obj_dict = homan.forward_obj_pose_render(loss_only=True, func='iou')  # (B,N)
            l_obj_mask = l_obj_dict['mask']
            l_obj_offscreen = l_obj_dict['offscreen']

            tot_loss = l_sil_hand.sum() + \
                (l_obj_mask.sum() + l_obj_offscreen.sum()) + \
                l_pca + l_hand_pose
                # l_min_d
            
            # print(
            #     # f"min_d:{l_min_d.item():.3f}"
            #     f"pca:{l_pca.item():.3f} "
            #     f"sil_hand:{l_sil_hand.sum().item():.3f} "
            #     f"obj_mask:{l_obj_mask.sum().item():.3f} "
            #     f"hand_pose:{l_hand_pose.item():.3f}")

            if vis_interval > 0 and step % vis_interval == 0:
                if writer is None:
                    _ = homan.render_grid(obj_idx=0, low_reso=True)
                else:
                    img = homan.render_grid_np(obj_idx=0)
                    writer.add_image(tag=f'{prefix}', 
                                     img_tensor=img.transpose(2, 0, 1),
                                     global_step=step)

                    scalar_dict = dict()
                    scalar_dict['tot_loss'] = tot_loss.item()
                    for b in range(homan.bsize):
                        scalar_dict.update({
                            f'hand_{b}': l_sil_hand[b].item(),
                            f'obj_{b}': l_obj_mask[b].item() + l_obj_offscreen[b].item()
                        })
                    writer.add_scalars(
                        main_tag=f'{prefix}', global_step=step,
                        tag_scalar_dict=scalar_dict)

            tot_loss.backward()
            optimizer.step()
            loop.set_description(f"tot_loss: {tot_loss.item():.3g}")
            loop.update()

    if writer is not None:
        writer.add_figure(tag=f'{prefix}_final', 
                        figure=homan.render_grid(),
                        global_step=step)
        writer.flush()
    
    return homan