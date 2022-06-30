import enum
import os.path as osp
import argparse
from PIL import Image

import numpy as np
import torch
from homan.ho_forwarder import HOForwarder


""" Run ihoi but with differentiable based pose optimizer. """


def get_args():
    parser = argparse.ArgumentParser(
        description="Simple Epic inference")
    parser.add_argument("input_dir", 
                        help="Path to directory where saved_context.pth"
                        "can be found")
    parser.add_argument("--clusters", type=int)

    args = parser.parse_args()
    return args


def optimize_scale(homan, num_steps=100) -> HOForwarder:
    scale_weights = dict(
        lw_pca=0.0,
        lw_collision=0.0,
        lw_contact=1.0,
        lw_sil_obj=1.0,
        lw_sil_hand=0.0,
        lw_inter=1.0,

        lw_scale_obj=0.0,  # mean deviation loss
        lw_scale_hand=0.0,
        lw_depth=1.0
    )

    lr = 1e-3
    optimizer = torch.optim.SGD([
        {
            'params': [homan.scale_object],
            'lr': lr
        }
    ])

    loop = range(num_steps)

    for step in loop:
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

        loss.backward()
        optimizer.step()

    return homan


def main(args):

    saved_context = osp.join(args.input_dir, 'saved_context.pth')
    ctx = torch.load(saved_context)
    num_clusters = args.clusters

    if num_clusters > 0:
        obj_pose_results = ctx.pose_machine.pose_model.clustered_results(
            K=num_clusters)
    else:
        obj_pose_results = ctx.pose_machine.pose_model.fitted_results
    ious = obj_pose_results.iou

    for ind, iou in enumerate(ious):
        print(f"Processing sorted ind={ind}, iou={iou}")
        homan_kwargs = HOForwarder.pack_homan_kwargs(
            ctx, ind, obj_pose_results=obj_pose_results)
        torch.cuda.empty_cache()

        homan = HOForwarder(**homan_kwargs).cuda()
        homan = optimize_scale(homan)
        img_triview = np.uint8(homan.render_summary())
        if num_clusters > 0:
            img_name = f"clu_{ind}_{int(iou*100):03d}.jpg"
        else:
            img_name = f"{ind}_{int(iou*100):03d}.jpg"
        Image.fromarray(img_triview).save(osp.join(args.input_dir, img_name))


if __name__ == "__main__":
    main(get_args())
