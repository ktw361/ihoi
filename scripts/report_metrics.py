import os
from typing import Union
import argparse
import tqdm
import numpy as np
import torch
import pandas as pd
from homan.ho_forwarder_v2 import HOForwarderV2Vis
from homan.mvho_forwarder import MVHOVis

import sys
sys.path.append('/home/skynet/Zhifan/repos/CPF')
from hocontact.utils.libmesh.inside_mesh import check_mesh_contains


""" This assume the output is in the format of:
vid_start_end_post.pth
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    args = parser.parse_args()
    return args


def max_intersect_volume(homan: Union[HOForwarderV2Vis, MVHOVis],
                         kind,
                         pitch=0.005, 
                         ret_all=False) -> float:
    """ Max Iv of object into hand, report in cm^3
    pitch: voxel size, 0.01m == 1cm
    """
    obj_idx = 0
    max_iv = 0
    T = homan.get_verts_object().shape[0]
    vox_size = np.power(pitch * 100, 3)
    iv_list = []
    for t in range(T):
        if kind == 'hov2':
            mhand, mobj = homan.get_meshes(t, obj_idx)
        elif kind == 'mvho':
            mhand, mobj = homan.get_meshes(0, t)

        obj_pts = mobj.voxelized(pitch=pitch).points
        inside = check_mesh_contains(mhand, obj_pts)
        volume = inside.sum() * vox_size
        iv_list.append(volume)
        max_iv = max(max_iv, volume)
    if ret_all:
        return iv_list
    else:
        return max_iv


def main(args):
    video_dir = args.dir

    vid_keys = []
    hious = []
    oious = []
    pds = []
    ivs = []
    pre_hious = []
    pre_oious = []
    pre_pds = []
    pre_ivs = []
    post_paths = sorted([v for v in os.listdir(video_dir) if v.endswith('_post.pth')])
    for ii, post_path in tqdm.tqdm(enumerate(post_paths), total=len(post_paths)):
        if not post_path.endswith('_post.pth'):
            continue
        vid_key = post_path.replace('_post.pth', '')

        pre_path = post_path.replace('_post.pth', '_model.pth')
        mvho = torch.load(os.path.join(video_dir, pre_path))
        hov2 = torch.load(os.path.join(video_dir, post_path))
        with torch.no_grad():
            pre_metrics = mvho.eval_metrics(
                unsafe=True, avg=True, post_homan=hov2)
            metrics = hov2.eval_metrics(unsafe=True, avg=True)
        pre_iv = max_intersect_volume(mvho, kind='mvho', pitch=0.005)
        iv = max_intersect_volume(hov2, kind='hov2', pitch=0.005)
        vid_keys.append(vid_key)
        pre_hious.append(pre_metrics['hious'])
        pre_oious.append(pre_metrics['oious'])
        pre_pds.append(pre_metrics['pd_h2o'])
        pre_ivs.append(pre_iv)

        hious.append(metrics['hious'])
        oious.append(metrics['oious'])
        pds.append(metrics['pd_h2o'])
        ivs.append(iv)

        # ckpt
        if ii % 50 == 0:
            df_pre = pd.DataFrame({'vid_key': vid_keys, 'hious': pre_hious, 'oious': pre_oious, 'pds': pre_pds, 'ivs': pre_ivs})
            df_pre.to_csv(os.path.join(video_dir, 'pre_metrics.csv'), index=False)
            df = pd.DataFrame({'vid_key': vid_keys, 'hious': hious, 'oious': oious, 'pds': pds, 'ivs': ivs})
            df.to_csv(os.path.join(video_dir, 'metrics.csv'), index=False)

    df_pre = pd.DataFrame({'vid_key': vid_keys, 'hious': pre_hious, 'oious': pre_oious, 'pds': pre_pds, 'ivs': pre_ivs})
    df_pre.to_csv(os.path.join(video_dir, 'pre_metrics.csv'), index=False)
    df = pd.DataFrame({'vid_key': vid_keys, 'hious': hious, 'oious': oious, 'pds': pds, 'ivs': ivs})
    df.to_csv(os.path.join(video_dir, 'metrics.csv'), index=False)


if __name__ == '__main__':
    main(parse_args())