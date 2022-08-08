from typing import Union
import torch
from homan.ho_forwarder import HOForwarder
from homan.hand_forwarder import HandForwarder



""" Different HO optimization plans. """


def optimize_hand(homan: HandForwarder,
                  lr=1e-2,
                  num_steps=100,
                  verbose=True):
    optimizer = torch.optim.Adam([
        {
            'params': [homan.rotations_hand, 
                       homan.translations_hand,
                       homan.mano_pca_pose],
            'lr': lr
        }
    ])

    for step in range(num_steps):
        optimizer.zero_grad()
        tot_loss, loss_dict = homan.forward()
        # loss = loss_dict['loss_sil_hand'].sum()
        if verbose and step % 10 == 0:
            print(f"Step {step}, total loss = {tot_loss.item():.05f}: ", end='\n')
            # print(iou)

        tot_loss.backward()
        optimizer.step()

    return homan


def optimize_homan_hand(homan: HOForwarder,
                        lr=1e-2,
                        num_steps=100,
                        verbose=True):
    optimizer = torch.optim.Adam([
        {
            'params': [homan.rotations_hand, 
                       homan.translations_hand,
                       homan.mano_pca_pose],
            'lr': lr
        }
    ])

    for step in range(num_steps):
        optimizer.zero_grad()
        loss_dict, iou = homan.forward_sil_hand()
        loss = loss_dict['loss_sil_hand'].sum()
        if verbose and step % 10 == 0:
            print(f"Step {step}, total loss = {loss:.05f}: ", end='\n')
            print(iou)

        loss.backward()
        optimizer.step()

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