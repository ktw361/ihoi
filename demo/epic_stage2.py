import os
import os.path as osp
import argparse
from PIL import Image

import numpy as np
import torch
from homan.ho_forwarder import HOForwarder
from libzhifan import io


""" Run ihoi but with differentiable based pose optimizer. """


def get_args():
    parser = argparse.ArgumentParser(
        description="Simple Epic inference")
    parser.add_argument("input_dir",
                        help="Path to directory where saved_context.pth"
                        "can be found")
    parser.add_argument('--compute_clusters', action='store_true')
    parser.add_argument('--no-compute_clusters', dest='compute_clusters',
                        action='store_false')
    parser.set_defaults(compute_clusters=False)
    parser.add_argument("--clusters", default=10, type=int)
    parser.add_argument("--select", default=-1, type=int,
                        help="The cluster center to run")

    args = parser.parse_args()
    return args


def optimize_scale(homan, num_steps=100) -> HOForwarder:
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

    lr = 1e-2
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
        if step % 10 == 0:
            print(f"Step {step}, total loss = {loss.item():.05f}: ", end='')
            for k, v in loss_dict_weighted.items():
                print(f"{k.replace('loss_', '')}, {v.item():.05f}", end=', ')
            print('\n', end='')
            print(f"scale={homan.scale_object.item()}")

        loss.backward()
        # print(f"grad_s: {homan.scale_object.grad.item()}")
        optimizer.step()

    loss_dict, metric_dict = homan(loss_weights=dict(
        lw_pca=0.0,
        lw_collision=1.0,
        lw_contact=1.0,
        lw_sil_obj=1.0,
        lw_sil_hand=0.0,
        lw_inter=1.0,

        lw_scale_obj=0.0,  # mean deviation loss
        lw_scale_hand=0.0,
        lw_depth=1.0
    ))
    return homan, loss_dict, metric_dict


def main(args):

    saved_context = osp.join(args.input_dir, 'saved_context.pth')
    ctx = torch.load(saved_context)
    num_clusters = args.clusters
    compute_clusters = args.compute_clusters

    if num_clusters > 0:
        save_dir = args.input_dir
        if compute_clusters:
            obj_pose_results = ctx.pose_machine.pose_model.clustered_results(
                K=num_clusters)
            torch.save(obj_pose_results, osp.join(save_dir, 'clustered_results.pt'))
        else:
            obj_pose_results = torch.load(osp.join(save_dir, 'clustered_results.pt'))
    else:
        obj_pose_results = ctx.pose_machine.pose_model.fitted_results
        save_dir = osp.join(args.input_dir, 'full')
        os.makedirs(save_dir, exist_ok=True)

    if args.select >= 0:
        ind = args.select
        iou = obj_pose_results.iou[ind]
        print(f"Processing sorted ind={ind}, iou={iou}")
        homan_kwargs = HOForwarder.pack_homan_kwargs(
            ctx, ind, obj_pose_results=obj_pose_results)
        torch.cuda.empty_cache()

        homan = HOForwarder(**homan_kwargs).cuda()
        homan, loss_dict, metric_dict = optimize_scale(homan)
        img_triview = np.uint8(homan.render_summary())
        if num_clusters > 0:
            img_name = f"clu_{ind}_{int(iou*100):03d}.jpg"
        else:
            img_name = f"{ind}_{int(iou*100):03d}.jpg"
        Image.fromarray(img_triview).save(osp.join(save_dir, img_name))
        return

    infos = dict()
    ious = obj_pose_results.iou
    for ind, iou in enumerate(ious):
        print(f"Processing sorted ind={ind}, iou={iou}")
        homan_kwargs = HOForwarder.pack_homan_kwargs(
            ctx, ind, obj_pose_results=obj_pose_results)
        torch.cuda.empty_cache()

        homan = HOForwarder(**homan_kwargs).cuda()
        homan, loss_dict, metric_dict = optimize_scale(homan)
        img_triview = np.uint8(homan.render_summary())
        if num_clusters > 0:
            img_name = f"clu_{ind}_{int(iou*100):03d}.jpg"
        else:
            img_name = f"{ind}_{int(iou*100):03d}.jpg"
        Image.fromarray(img_triview).save(osp.join(save_dir, img_name))

        # Save results
        entry = [
            homan.scale_object.item(),
            loss_dict['loss_contact'].item(),
            loss_dict['loss_sil_obj'].item(),
            loss_dict['loss_collision'].item(),
            loss_dict['loss_inter'].item(),
            loss_dict['loss_depth'].item(),
            metric_dict['iou_object'],
            metric_dict['handobj_maxdist'],
        ]
        infos[img_name] = entry

    io.write_json(infos, osp.join(save_dir, 'infos.json'), indent=2)


if __name__ == "__main__":
    main(get_args())
