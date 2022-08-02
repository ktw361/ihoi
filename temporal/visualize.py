from typing import List
import matplotlib.pyplot as plt
import numpy as np
from obj_pose.pose_optimizer import PoseOptimizer

from libzhifan.geometry import visualize_mesh, SimpleMesh


def plot_summaries(homans) -> plt.figure:
    """ homans: list of HO_forwarder """
    l = len(homans)
    num_cols = 5
    num_rows = l // num_cols + 1
    fig, axes = plt.subplots(
        nrows=num_rows, ncols=num_cols,
        sharex=True, sharey=True, figsize=(20, 20))
    idx = 0
    for idx, ax in enumerate(axes.flat, start=0):
        homan = homans[idx]
        img = homan.render_summary()
        ax.imshow(img)
        ax.set_axis_off()
        idx += 1
        if idx >= l:
            break

    plt.tight_layout()
    return fig


def plot_pose_summaries(pose_machine: PoseOptimizer,
                        pose_idx=0) -> plt.figure:
    """ homans: list of HO_forwarder """
    l = len(pose_machine.global_cam)
    num_cols = 5
    num_rows = (l + num_cols - 1) // num_cols
    fig, axes = plt.subplots(
        nrows=num_rows, ncols=num_cols,
        sharex=True, sharey=True, figsize=(20, 20))
    for cam_idx, ax in enumerate(axes.flat, start=0):
        img = pose_machine.render_model_output(
            pose_idx, cam_idx=cam_idx, kind='ihoi',
            with_obj=True)
        ax.imshow(img)
        ax.set_axis_off()
        if cam_idx == l-1:
            break

    plt.tight_layout()
    return fig


def concat_pose_meshes(pose_machine: PoseOptimizer,
                       pose_idx=0,
                       obj_file=None):
    """
    Returns a list of SimpleMesh,
    offset each timestep for easier comparison?
    """
    meshes = []
    l = len(pose_machine.global_cam)
    obj_verts = pose_machine.pose_model.fitted_results.verts
    disp = 0.15  # displacement
    for cam_idx in range(l):
        hand_mesh = pose_machine.hand_simplemesh(cam_idx=cam_idx)
        obj_mesh = SimpleMesh(
            obj_verts[cam_idx, pose_idx],
            pose_machine.pose_model.faces,
            tex_color='yellow')
        hand_mesh.apply_translation_([cam_idx * disp, 0, 0])
        obj_mesh.apply_translation_([cam_idx * disp, 0, 0])
        meshes.append(hand_mesh)
        meshes.append(obj_mesh)

    if obj_file is not None:
        visualize_mesh(meshes, show_axis=False).export(
            obj_file)
    return meshes
