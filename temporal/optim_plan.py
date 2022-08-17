from typing import Union
import tqdm
import numpy as np
import torch
from homan.ho_forwarder_v2 import HOForwarderV2Impl


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
                          debug=True,
                          viz_folder='output/tmp',
                          viz_step=10,
                          sort_best=True) -> HOForwarderV2Impl:
    device = 'cuda'
    optimizer = torch.optim.Adam([
        {
            'params': [
                homan.rotations_object,
                homan.translations_object,
            ],
            'lr': lr
        }
    ])

    best_losses = torch.tensor(np.inf)
    best_rots = None
    best_trans = None
    best_loss_single = torch.tensor(np.inf)
    best_trans_single = None
    loop = tqdm.tqdm(total=num_iterations)

    for _ in range(num_iterations):
        optimizer.zero_grad()
        loss_dict, _, sil = homan.forward_obj_pose_render()
        # if debug and (step % viz_step == 0):
        #     mask_viz = mask[0]  # select 0-th mask for visualization
        #     debug_viz_folder = os.path.join(viz_folder, "poseoptim")
        #     os.makedirs(debug_viz_folder, exist_ok=True)
        #     imagify.viz_imgrow(
        #         sil[0], overlays=[mask_viz,]*len(sil[0]), viz_nb=4,
        #         path=os.path.join(debug_viz_folder, f"{step:04d}.png"))

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
    if best_rots is None:
        best_rots = homan.rotations_object
        best_trans = homan.translations_object
        best_losses = losses
    else:
        best_rots = torch.cat((best_rots, homan.rotations_object), 0)
        best_trans = torch.cat((best_trans, homan.translations_object), 0)
        best_losses = torch.cat((best_losses, losses))
    if sort_best:
        inds = torch.argsort(best_losses)
        num_obj_init = homan.num_obj_init
        best_losses = best_losses[inds][:num_obj_init].detach().clone()
        best_trans = best_trans[inds][:num_obj_init].detach().clone()
        best_rots = best_rots[inds][:num_obj_init].detach().clone()
    loop.close()
    # Add best ever:

    if sort_best:
        best_rots = torch.cat((best_rots_single.unsqueeze(0), best_rots[:-1]),
                              0)
        best_trans = torch.cat(
            (best_trans_single.unsqueeze(0), best_trans[:-1]), 0)
    # model.rotations = nn.Parameter(best_rots)
    # model.translations = nn.Parameter(best_trans)
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