from typing import NamedTuple
import os
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import trimesh
from pytorch3d.transforms import Transform3d

from nnutils import image_utils, geom_utils
from nnutils.handmocap import (
    get_hand_wrapper, compute_hand_transform, cam_from_bbox)
from obj_pose.pose_renderer import PoseRenderer
from homan.lib3d.optitrans import TCO_init_from_boxes_zup_autodepth
from homan.utils.geometry import (
    compute_random_rotations,
    matrix_to_rot6d,
)
from temporal.utils import init_6d_pose_from_bboxes

from libzhifan.geometry import (
    SimpleMesh, projection
)
from libyana.lib3d import kcrop
from libyana.visutils import imagify


from config.epic_constants import REND_SIZE

"""
Refinement-based pose optimizer.

Origin: homan/pose_optimization.py
"""


class PoseOptimizer:

    NUM_CLUSTERS = 10

    """
    This class 1) parses mocap_prediction, and 2) performs obj pose fitting.

    Note: Bounding boxes mentioned in the pipeline:

    `hand_bbox`, `obj_bbox`: original boxes labels in epichoa
    `hand_bbox_proc`: processed hand_bbox during mocap regressing
    `obj_bbox_squared`: squared obj_bbox before diff rendering

    """

    def __init__(self,
                 one_hand,
                 obj_loader,
                 side,
                 ihoi_box_expand=1.0,
                 rend_size=REND_SIZE,
                 device='cuda',
                 hand_rotation=None,
                 hand_translation=None,
                 ):
        """
        Args:
            hand_wrapper: Non flat ManoWrapper
                e.g.
                    hand_wrapper = ManopthWrapper(
                        flat_hand_mean=False,
                        use_pca=False,
                        )

        """
        self.obj_loader = obj_loader
        self.hand_wrapper = get_hand_wrapper(side)
        self.rend_size = rend_size
        self.ihoi_box_expand = ihoi_box_expand
        self.device = device

        # Process one_hand
        _pred_hand_pose, _pred_hand_betas, pred_camera = map(
            lambda x: torch.as_tensor(one_hand[x], device=device),
            ('pred_hand_pose', 'pred_hand_betas', 'pred_camera'))
        _hand_bbox_proc = one_hand['bbox_processed']
        rot_axisang, _pred_hand_pose = _pred_hand_pose[:, :3], _pred_hand_pose[:, 3:]

        if hand_rotation is not None and hand_translation is not None:
            self._hand_rotation = hand_rotation
            self._hand_translation = hand_translation
        else:
            self._hand_rotation, self._hand_translation = compute_hand_transform(
                rot_axisang, _pred_hand_pose, pred_camera, side=side)
        _hand_verts = self._calc_hand_mesh(_pred_hand_pose)
        hand_cam, global_cam = cam_from_bbox(_hand_bbox_proc)

        self._pred_hand_pose = _pred_hand_pose  # (B, 45)
        self._pred_hand_betas = _pred_hand_betas
        self._hand_verts = _hand_verts

        self._hand_cam = hand_cam
        self._hand_bbox_proc = _hand_bbox_proc
        self.global_cam = global_cam

        """ placeholder for cache after fitting """
        self._fit_model = None
        self._full_image = None

    def __len__(self):
        return len(self.global_cam)

    @property
    def pred_hand_pose(self):
        return self._pred_hand_pose

    @property
    def pred_hand_betas(self):
        return self._pred_hand_betas

    @property
    def hand_faces(self):
        """ always (1, F, 3) """
        return self.hand_wrapper.hand_faces

    @property
    def hand_verts(self):
        return self._hand_verts

    @property
    def hand_cam(self):
        return self._hand_cam

    @property
    def pose_model(self):
        if self._fit_model is None:
            raise ValueError("model not fitted yet.")
        else:
            return self._fit_model

    @property
    def hand_rotation(self):
        return self._hand_rotation

    @property
    def hand_translation(self):
        """ (B, 1, 3) """
        return self._hand_translation

    def hand_simplemesh(self, cam_idx: int = 0):
        return SimpleMesh(
            self.hand_verts[cam_idx], self.hand_faces[0])

    def _calc_hand_mesh(self,
                        pred_hand_pose):
        """

        When Ihoi predict the MANO params,
        it takes the `hand_bbox_list` from dataset,
        then pad_resize the bbox, which results in the `bbox_processed`.

        The MANO params are predicted in
            global_cam.crop(hand_bbox).resize(224, 224),
        so we need to emulate this process, see CameraManager below.

        Args:
            pred_hand_pose: (B, 45)

        Returns:
            hand_verts: (B, V, 3) torch.Tensor
            hand_verts_transformed: (B, V, 3) torch.Tensor
        """
        v_orig, _, _, _ = self.hand_wrapper(
            None, pred_hand_pose, mode='inner', return_mesh=False)

        cTh = geom_utils.rt_to_homo(
            self.hand_rotation.transpose(1, 2), t=self.hand_translation.squeeze(1))
        cTh = Transform3d(matrix=cTh.transpose(1, 2))
        v_transformed = cTh.transform_points(v_orig)
        return v_transformed

    def render_model_output(self,
                            pose_idx: int,
                            cam_idx: int = 0,
                            kind: str='ihoi',
                            clustered=True,
                            with_hand=True,
                            with_obj=True):
        """
        Args:
            idx: index into model.apply_transformation()
            kind: str, one of
                - 'global': render w.r.t full image
                - 'ihoi': render w.r.t image prepared
                    according to process_mocap_predictions()
        """
        hand_mesh = self.hand_simplemesh(cam_idx=cam_idx)
        if clustered:
            verts = self.pose_model.clustered_results(
                self.NUM_CLUSTERS).verts
        else:
            verts = self.pose_model.fitted_results.verts
        verts = verts[cam_idx, pose_idx]
        obj_mesh = SimpleMesh(verts,
                              self.pose_model.faces,
                              tex_color='yellow')
        mesh_list = []
        if with_hand:
            mesh_list.append(hand_mesh)
        if with_obj:
            mesh_list.append(obj_mesh)
        if kind == 'global':
            img = projection.perspective_projection_by_camera(
                mesh_list,
                self.global_cam[cam_idx],
                method=dict(
                    name='pytorch3d',
                    coor_sys='nr',
                    in_ndc=False
                ),
                image=self._full_image[cam_idx]
            )
            return img
        elif kind == 'ihoi':
            img = projection.perspective_projection_by_camera(
                mesh_list,
                self.ihoi_cam[cam_idx],
                method=dict(
                    name='pytorch3d',
                    coor_sys='nr',
                    in_ndc=False
                ),
                image=np.uint8(self._image_patch[cam_idx]),
            )
            return img
        elif kind == 'none':
            return np.uint8(self._image_patch[cam_idx])
        else:
            raise ValueError(f"kind {kind} not unserstood")

    def to_scene(self,
                 pose_idx: int,
                 cam_idx: int = 0,
                 clustered=True,
                 show_axis=True) -> trimesh.scene.Scene:
        """
        Args:
            idx: index into model.apply_transformation()
            kind: str, one of
                - 'global': render w.r.t full image
                - 'ihoi': render w.r.t image prepared
                    according to process_mocap_predictions()
        """
        from libzhifan.geometry import SimpleMesh, visualize_mesh
        hand_mesh = self.hand_simplemesh(cam_idx=cam_idx)
        if clustered:
            verts = self.pose_model.clustered_results(
                self.NUM_CLUSTERS).verts
        else:
            verts = self.pose_model.fitted_results.verts
        verts = verts[cam_idx, pose_idx]
        obj_mesh = SimpleMesh(verts,
                              self.pose_model.faces,
                              tex_color='yellow')
        return visualize_mesh([hand_mesh, obj_mesh],
                              show_axis=show_axis,
                              viewpoint='nr')

    def finalize_without_fit(self,
                             image,
                             obj_bbox,
                             object_mask):
        """
        Args:
            image: np.ndarray (B, H, W, 3)
            object_mask: torch.Tensor (B, H, W)
            obj_bbox: torch.Tensor (B, 4)

        Returns:
            obj_bbox_squared: torch.Tensor (B, 4)
            image_patch: np.ndarray (B, h, w, 3)
            obj_mask_patch: torch.Tensor (B, h, w)

        Side Effect:
            self._image_patch: (B, h, w, 3) ndarray
            self._full_image: (B, H, W, 3)
            self.ihoi_cam: BatchCameraManager
        """
        obj_bbox_squared = image_utils.square_bbox_xywh(
            obj_bbox, self.ihoi_box_expand).int()

        bsize = len(image)
        obj_mask_patch = image_utils.batch_crop_resize(
            object_mask, obj_bbox_squared, self.rend_size)

        # Extra care for image_patch with (H, W, 3) shape
        image_patch = image_utils.batch_crop_resize(
            image, obj_bbox_squared, self.rend_size)

        """ Get camera """
        global_cam = self.global_cam

        """ Apply the side effect """
        self._image_patch = image_patch
        self._full_image = image
        ihoi_h = torch.ones([bsize]) * self.rend_size
        ihoi_w = torch.ones([bsize]) * self.rend_size
        self.ihoi_cam = global_cam.crop(obj_bbox_squared).resize(
            ihoi_h, ihoi_w)

        return obj_bbox_squared, image_patch, obj_mask_patch

    def fit_obj_pose(self,
                     image,
                     obj_bbox,
                     object_mask,
                     cat,

                     rotations_init=None,
                     translations_init=None,
                     num_initializations=200,
                     num_iterations=50,
                     debug=True,
                     sort_best=False,
                     viz=True,
                     lr=1e-2,

                     put_hand_transform=False,
                     ):
        """
        Args: See EpicInference dataset output.
            image: (B, H, W, 3) torch.Tensor. possibly (720, 1280, 3)
            obj_bbox: (B, 4)
            object_mask: (B, H, W) int ndarray of (-1, 0, 1)
            cat: str
            hand_bbox:processed: (B, 4)
                Note this differs from `hand_bbox` directly from dataset

        Returns:
            PoseRenderer
        """
        obj_bbox_squared, _, obj_mask_patch = self.finalize_without_fit(
            image, obj_bbox, object_mask)

        obj_mesh = self.obj_loader.load_obj_by_name(cat, return_mesh=False)
        vertices = torch.as_tensor(obj_mesh.vertices, device='cuda')
        faces = torch.as_tensor(obj_mesh.faces, device='cuda')

        if rotations_init is not None and translations_init is not None:
            assert len(rotations_init) == len(translations_init)
            num_initializations = len(rotations_init)

        base_rotation = None
        base_translation = None
        if put_hand_transform:
            base_rotation = self.hand_rotation
            base_translation = self.hand_translation
        model = find_optimal_pose(
            vertices=vertices,
            faces=faces,
            bbox=obj_bbox,
            mask=obj_mask_patch,
            K_global=self.global_cam.get_K(),
            K_ihoi_nr=self.ihoi_cam.to_nr(return_mat=True),

            rotations_init=rotations_init,
            translations_init=translations_init,
            base_rotation=base_rotation,
            base_translation=base_translation,
            num_initializations=num_initializations,
            num_iterations=num_iterations,
            debug=debug,
            sort_best=sort_best,
            lr=lr,
        )

        self._fit_model = model
        return self._fit_model


def find_optimal_pose(
    vertices,
    faces,
    mask,
    bbox,
    K_global,
    K_ihoi_nr,
    num_iterations=50,
    num_initializations=2000,
    lr=1e-2,
    debug=True,
    viz_folder="output/tmp",
    viz_step=10,
    sort_best=True,
    rotations_init=None,
    translations_init=None,
    base_rotation=None,
    base_translation=None):
    """
    Args:
        vertices: torch.Tensor (V, 3)
        faces: torch.Tensor (F, 3)
        mask: troch.Tensor (B, W, W)
            1 for fg, 0 for bg, -1 for ignored
        bbox: torch.Tensor (B, 4)
            XYWH bbox from original data source. E.g. epichoa/mask_gen
        K_global: torch.Tensor (B, 3, 3), represents the global camera that
            captures the original image.
        K_ihoi_nr: torch.Tensor (B, 3, 3), represents the global camera that
            captures the original image.
        base_rotation: torch.Tensor (B, 3, 3)
        base_translation: torch.Tensor (B, 1, 3)

    Returns:
        A PoseRenderer Object,
            - __call__():
                returns loss_dict, iou, image
            - render():
                returns image
            - apply_transformation():
                returns: R @ V + T
            - rotation
            - translations
            - K
    """
    bsize = bbox.size(0)
    os.makedirs(viz_folder, exist_ok=True)
    device = vertices.device

    # Stuff to keep around
    best_losses = torch.tensor(np.inf)
    best_rots = None
    best_trans = None
    best_loss_single = torch.tensor(np.inf)
    best_rots_single = None
    best_trans_single = None
    loop = tqdm(total=num_iterations)
    K_global = K_global.to(device)
    K_ihoi_nr = K_ihoi_nr.to(device)
    # Mask is in format 256 x 256 (REND_SIZE x REND_SIZE)
    # bbox is in xywh in original image space
    # K is in pixel space
    # If initial rotation is not provided, it is sampled
    # uniformly from SO3
    if rotations_init is None:
        rotations_init = compute_random_rotations(
            num_initializations, upright=False, device=device)

    # Translation is retrieved by matching the tight bbox of projected
    # vertices with the bbox of the target mask
    if base_rotation is None:
        base_rotation = torch.eye(
            3, dtype=rotations_init.dtype, device=device).unsqueeze_(0)
        base_rotation = base_rotation.repeat(bsize, 1, 1)
    else:
        base_rotation = base_rotation.clone()
    if base_translation is None:
        base_translation = torch.zeros(
            [bsize, 1, 3], dtype=rotations_init.dtype, device=device)
    else:
        base_translation = base_translation.clone()

    rotations_init, translations_init = \
        init_6d_pose_from_bboxes(
            bbox, vertices, cam_mat=K_global,
            num_init=num_initializations,
            base_rotation=base_rotation,
            base_translation=base_translation)

    model = PoseRenderer(
        ref_image=mask,
        vertices=vertices,
        faces=faces,
        rotation_init=matrix_to_rot6d(rotations_init),
        translation_init=translations_init,
        num_initializations=num_initializations,
        camera_K=K_ihoi_nr,
        base_rotation=base_rotation,
        base_translation=base_translation
    )
    # model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(num_iterations):
        optimizer.zero_grad()
        loss_dict, _, sil = model()
        if debug and (step % viz_step == 0):
            mask_viz = mask[0]  # select 0-th mask for visualization
            debug_viz_folder = os.path.join(viz_folder, "poseoptim")
            os.makedirs(debug_viz_folder, exist_ok=True)
            imagify.viz_imgrow(
                sil[0], overlays=[mask_viz,]*len(sil[0]), viz_nb=4,
                path=os.path.join(debug_viz_folder, f"{step:04d}.png"))

        losses = sum(loss_dict.values())
        loss = losses.sum()
        loss.backward()
        optimizer.step()
        if losses.min() < best_loss_single:
            ind = torch.argmin(losses)
            best_loss_single = losses[ind]
            best_rots_single = model.rotations[ind].detach().clone()
            best_trans_single = model.translations[ind].detach().clone()
        loop.set_description(f"obj loss: {best_loss_single.item():.3g}")
        loop.update()
    if best_rots is None:
        best_rots = model.rotations
        best_trans = model.translations
        best_losses = losses
    else:
        best_rots = torch.cat((best_rots, model.rotations), 0)
        best_trans = torch.cat((best_trans, model.translations), 0)
        best_losses = torch.cat((best_losses, losses))
    if sort_best:
        inds = torch.argsort(best_losses)
        best_losses = best_losses[inds][:num_initializations].detach().clone()
        best_trans = best_trans[inds][:num_initializations].detach().clone()
        best_rots = best_rots[inds][:num_initializations].detach().clone()
    loop.close()
    # Add best ever:

    if sort_best:
        best_rots = torch.cat((best_rots_single.unsqueeze(0), best_rots[:-1]),
                              0)
        best_trans = torch.cat(
            (best_trans_single.unsqueeze(0), best_trans[:-1]), 0)
    model.rotations = nn.Parameter(best_rots)
    model.translations = nn.Parameter(best_trans)
    return model


class SavedContext(NamedTuple):
    pose_machine: PoseOptimizer
    obj_bbox: np.ndarray
    mask_hand: np.ndarray
    mask_obj: np.ndarray
    hand_side: str
