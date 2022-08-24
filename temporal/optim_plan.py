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

    loss_weights = {
        'sil': 1,
        'pca': 0,
        'rot': 1,
        'transl': 1,
    }
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
                          num_iterations=50,
                          lr=1e-2,
                          sort_best=True) -> HOForwarderV2Impl:
    optimizer = torch.optim.Adam([
        {
            'params': [
                homan.rotations_hand,  # 
                homan.translations_hand,  #
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
            loss_dict = homan.forward_obj_pose_render(loss_only=True)

            losses = sum(loss_dict.values())
            loss = losses.sum()
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
                          verbose=True,
                          vis_interval=-1):
    optimizer = torch.optim.Adam([
        {
            'params': [
                homan.rotations_hand,
                # homan.translations_hand,
                # homan.rotations_object,
                homan.translations_object,
                homan.scale_object,
            ],
            'lr': lr
        }
    ])

    with tqdm.tqdm(total=num_steps) as loop:
        for step in range(num_steps):
            optimizer.zero_grad()
            l_sil_hand = homan.loss_sil_hand(compute_iou=False, func='l2')
            l_obj_dict = homan.forward_obj_pose_render(loss_only=True, func='l2')
            tot_loss = l_sil_hand.sum() + 0.1 * sum(l_obj_dict.values()).sum()

            if vis_interval > 0 and step % vis_interval == 0:
                _ = homan.render_grid(obj_idx=0, low_reso=True)

            tot_loss.backward()
            optimizer.step()
            loop.set_description(f"obj loss: {tot_loss.item():.3g}")
            loop.update()
    
    return homan

# def optimize_scale(homan,
#                    num_steps=100,
#                    lr=1e-2,
#                    verbose=False) -> HOForwarder:
#     scale_weights = dict(
#         lw_pca=0.0,
#         lw_collision=1.0,
#         lw_contact=1.0,
#         lw_sil_obj=1.0,
#         lw_sil_hand=0.0,
#         lw_inter=1.0,

#         lw_scale_obj=0.0,  # mean deviation loss
#         lw_scale_hand=0.0,
#         lw_depth=1.0
#     )

#     optimizer = torch.optim.Adam([
#         {
#             'params': [homan.scale_object],
#             'lr': lr
#         }
#     ])

#     for step in range(num_steps):
#         optimizer.zero_grad()
#         loss_dict, metric_dict = homan(loss_weights=scale_weights)
#         loss_dict_weighted = {
#             k: loss_dict[k] * scale_weights[k.replace("loss", "lw")]
#             for k in loss_dict
#         }
#         loss = sum(loss_dict_weighted.values())
#         if verbose and step % 10 == 0:
#             print(f"Step {step}, total loss = {loss.item():.05f}: ", end='')
#             for k, v in loss_dict_weighted.items():
#                 print(f"{k.replace('loss_', '')}, {v.item():.05f}", end=', ')
#             print('\n', end='')
#             print(f"scale={homan.scale_object.item()}")

#         loss.backward()
#         optimizer.step()

#     return homan