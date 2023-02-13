from collections import namedtuple
from typing import NamedTuple
import tqdm
import torch
from homan.ho_forwarder_v2 import HOForwarderV2Vis, HOForwarderV2Impl
from temporal.utils import choose_with_softmax

from moviepy import editor


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


def reinit_sample_optimize(homan: HOForwarderV2Vis,
                           rotation6d_inits,
                           translation_inits,
                           scale_inits,
                           weights=None,
                           save_grid: str = None,
                           cfg = None,
                           debug_num_epoch=None):
    """
    Args:
        save_grid: str, fname to save the optimization process
        cfg: cfg.optim in config/conf.yaml
    """
    # Read out from config
    lr = cfg.lr
    num_epoch_parallel = cfg.num_epoch_parallel
    num_iters = cfg.num_iters
    temperature = cfg.temperature
    ratio = cfg.ratio
    vis_interval = cfg.vis_interval
    criterion = cfg.criterion

    ElementType = namedtuple(
        "ElementType", "iou collision min_dist R t s sample_indices")

    weights = homan.rotations_hand.new_zeros([homan.bsize]) \
        if weights is None else weights
    if save_grid:
        out_frames = []

    results = []

    num_epochs = rotation6d_inits.shape[0] if debug_num_epoch is None else debug_num_epoch
    for e in tqdm.trange(num_epochs // num_epoch_parallel, disable=not cfg.epoch_tqdm):

        homan.sample_indices = choose_with_softmax(
            weights, temperature=temperature, ratio=ratio)  # same for all num_epoch_parallel

        transform_indices = torch.arange(e*num_epoch_parallel, (e+1)*num_epoch_parallel)
        homan.set_obj_transform(
            translation_inits[transform_indices,...],
            rotation6d_inits[transform_indices,...],
            scale_inits[transform_indices,...])
        homan._check_shape_object(homan.num_obj)

        params = [
            homan.rotations_object,  # (1,)
            homan.translations_object,
            homan.scale_object,
        ]

        optimizer = torch.optim.Adam([{
            'params': params,
            'lr': lr
        }])

        with tqdm.tqdm(total=num_iters, disable=not cfg.iter_tqdm) as loop:
            for step in range(num_iters):
                optimizer.zero_grad()

                print_metric = (vis_interval > 0 and step % vis_interval == 0)
                tot_loss = homan.train_loss(
                    cfg=cfg, print_metric=print_metric)

                if save_grid and step % 5 == 0:
                    frame = homan.render_grid_np(0, True)
                    out_frames.append(frame)

                if torch.isnan(tot_loss) or torch.isinf(tot_loss):
                    tot_loss = weights.abs().max().detach().clone()
                    break
                tot_loss.backward()
                optimizer.step()
                loop.set_description(f"tot_loss: {tot_loss.item():.3g}")
                loop.update()

        with torch.no_grad():
            metrics = homan.eval_metrics()
            iou = metrics['iou']                # bigger better
            collision = metrics['collision']    # smaller better
            min_dist = metrics['min_dist']      # smaller better
            mean_iou = iou.mean(0)
            mean_collision = collision.mean(0)
            mean_min_dist = min_dist.mean(0)
            # v_hand = homan.get_verts_hand()[homan.sample_indices, ...]
            # v_obj = homan.get_verts_object()[homan.sample_indices, ...]
            # mask_score = homan.forward_obj_pose_render()['mask'].sum(0)
            # inside_score = homan.loss_insideness(
            #     v_hand=v_hand, v_obj=v_obj).sum(0)
            # close_score = homan.loss_closeness(
            #     v_hand=v_hand, v_obj=v_obj).sum(0)
            R = homan.rotations_object.detach().clone()
            t = homan.translations_object.detach().clone()
            s = homan.scale_object.detach().clone()
            for i in range(num_epoch_parallel):
                element = ElementType(
                    mean_iou[i].item(), mean_collision[i].item(), mean_min_dist[i].item(),
                    R[[i]], t[[i]], s[[i]], homan.sample_indices)
                results.append(element)
        # Update weights
        weights[homan.sample_indices] -= tot_loss  # TODO, is it good?

    if save_grid:
        editor.ImageSequenceClip(
            [v*255 for v in out_frames], fps=15).write_videofile(save_grid)

    # write-back best
    sign = 1 if criterion == 'iou' else -1
    final_score = \
        torch.softmax(torch.as_tensor([sign * getattr(v, criterion) for v in results]), 0)

    best_idx = final_score.argmax()
    best_metric = {
        'iou': results[best_idx].iou, 'collision': results[best_idx].collision,
        'min_dist': results[best_idx].min_dist}
    R, t, s = results[best_idx].R, results[best_idx].t, results[best_idx].s
    homan.set_obj_transform(
        translations_object=t,
        rotations_object=R,
        scale_object=s)
    homan.sample_indices = sorted(results[best_idx].sample_indices)
    homan._check_shape_object(1)

    return homan, weights, results, best_metric
