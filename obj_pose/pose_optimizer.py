from typing import NamedTuple
import os
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import trimesh

import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision import transforms
from pytorch3d.transforms import Transform3d
import pytorch3d.transforms.rotation_conversions as rot_cvt

from nnutils import image_utils, geom_utils
from obj_pose.pose_renderer import PoseRenderer, REND_SIZE

from libzhifan.geometry import (
    SimpleMesh, BatchCameraManager, projection
)
from libyana.camutils import project
from libyana.lib3d import kcrop
from libyana.visutils import imagify

from homan.lib3d.optitrans import (
    TCO_init_from_boxes_zup_autodepth,
)
from homan.utils.geometry import (
    compute_random_rotations,
    matrix_to_rot6d,
)

"""
Refinement-based pose optimizer.

Origin: homan/pose_optimization.py

"""


class PoseOptimizer:

    WEAK_CAM_FX = 10

    FULL_HEIGHT = 720
    FULL_WIDTH = 1280

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
                 hand_wrapper,
                 ihoi_box_expand=1.0,
                 rend_size=REND_SIZE,
                 device='cuda',
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
        self.one_hand = one_hand
        self.obj_loader = obj_loader
        self.hand_wrapper = hand_wrapper
        self.rend_size = rend_size
        self.ihoi_box_expand = ihoi_box_expand
        self.device = device

        # Process one_hand
        _pred_hand_pose, _pred_hand_betas, pred_camera = map(
            lambda x: torch.as_tensor(one_hand[x], device=device),
            ('pred_hand_pose', 'pred_hand_betas', 'pred_camera'))
        _hand_bbox_proc = one_hand['bbox_processed']
        rot_axisang, _pred_hand_pose = _pred_hand_pose[:, :3], _pred_hand_pose[:, 3:]

        self._hand_rotation, self._hand_translation = self._compute_hand_transform(
            rot_axisang, _pred_hand_pose, pred_camera)
        _hand_verts_orig, _hand_verts, hand_cam, global_cam = self._calc_hand_mesh(
            _pred_hand_pose, _pred_hand_betas,
            _hand_bbox_proc
        )

        self._pred_hand_pose = _pred_hand_pose  # (1, 45)
        self._pred_hand_betas = _pred_hand_betas
        self._hand_verts = _hand_verts
        self._hand_verts_orig = _hand_verts_orig

        self._hand_cam = hand_cam
        self._hand_bbox_proc = _hand_bbox_proc
        self.global_cam = global_cam

        """ placeholder for cache after fitting """
        self._fit_model = None
        self._full_image = None

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
    def hand_verts_orig(self):
        """ hand_verts before applying [self.hand_rotation | self.hand_translation] """
        return self._hand_verts_orig

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
        return self._hand_translation

    def hand_simplemesh(self, cam_idx: int = 0):
        return SimpleMesh(
            self.hand_verts[cam_idx], self.hand_faces[0])

    def recover_pca_pose(self):
        """
        if
            v_exp = ManopthWrapper(pca=False, flat=False).(x_0)
            x_pca = self.recover_pca_pose(self.x_0)  # R^45
        then
            v_act = ManoLayer(pca=True, flat=False, ncomps=45).forward(x_pca)
            v_exp == v_act

        note above requires mano_rot == zeros, since the computation of rotation
            is different in ManopthWrapper
        """
        M_pca_inv = torch.inverse(
            self.hand_wrapper.mano_layer_side.th_comps)
        mano_pca_pose = self.pred_hand_pose.mm(M_pca_inv)
        return mano_pca_pose

    def _compute_hand_transform(self, rot_axisang, pred_hand_pose, pred_camera):
        """
        Args:
            rot_axisang: (B, 3)
            pred_hand_pose: (B, 45)
            pred_camera: (B, 3)
                Used for translate hand_mesh to convert hand_mesh
                so that result in a weak perspective camera.

        Returns:
            rotation: (B, 3, 3) row-vec
            translation: (B, 3)
        """
        rotation = rot_cvt.axis_angle_to_matrix(rot_axisang)  # (1, 3) - > (1, 3, 3)
        rot_homo = geom_utils.rt_to_homo(rotation)
        glb_rot = geom_utils.matrix_to_se3(rot_homo)  # (1, 4, 4) -> (1, 12)
        _, joints = self.hand_wrapper(
            glb_rot,
            pred_hand_pose, return_mesh=True)
        fx = self.WEAK_CAM_FX
        s, tx, ty = torch.split(pred_camera, [1, 1, 1], dim=1)
        translate = torch.cat([tx, ty, fx/s], dim=1)
        translation = translate - joints[:, 5]
        rotation_row = rotation.transpose(1, 2)
        return rotation_row, translation

    def _calc_hand_mesh(self,
                        pred_hand_pose,
                        pred_hand_betas,
                        hand_bbox_proc,
                        ):
        """

        When Ihoi predict the MANO params,
        it takes the `hand_bbox_list` from dataset,
        then pad_resize the bbox, which results in the `bbox_processed`.

        The MANO params are predicted in
            global_cam.crop(hand_bbox).resize(224, 224),
        so we need to emulate this process, see CameraManager below.

        Args:
            pred_hand_pose: (B, 45)
            pred_hand_betas: Unused
            bbox_processed: (B, 4) bounding box for hand
                this box should be used in mocap_predictor.
                hand bounding box XYWH in original image
                same as one_hand['bbox_processed']

        Returns:
            hand_verts: (B, V, 3) torch.Tensor
            hand_verts_transformed: (B, V, 3) torch.Tensor
            hand_camera: CameraManager
            global_camera: CameraManager
        """
        v_orig, _, _, _ = self.hand_wrapper(
            None, pred_hand_pose, mode='inner', return_mesh=False)

        cTh = geom_utils.rt_to_homo(
            self.hand_rotation.transpose(1, 2), t=self.hand_translation)
        cTh = Transform3d(matrix=cTh.transpose(1, 2))
        v_transformed = cTh.transform_points(v_orig)

        _, _, hand_h, hand_w = torch.split(hand_bbox_proc, [1, 1, 1, 1], dim=1)
        hand_h = hand_h.view(-1)
        hand_w = hand_w.view(-1)
        hand_cam = self._hand_cam_from_bbox(hand_bbox_proc)
        global_cam = hand_cam.resize(hand_h, hand_w).uncrop(
            hand_bbox_proc, self.FULL_HEIGHT, self.FULL_WIDTH
        )
        return v_orig, v_transformed, hand_cam, global_cam

    def _hand_cam_from_bbox(self, hand_bbox) -> BatchCameraManager:
        """
        Args:
            hand_bbox: (B, 4) in global screen space
                possibly hand_bbox processed after mocap
        """
        hand_crop_h = 224
        hand_crop_w = 224
        fx = self.WEAK_CAM_FX
        _, _, box_w, box_h = torch.split(hand_bbox, [1, 1, 1, 1], dim=1)
        box_h = box_h.view(-1)
        box_w = box_w.view(-1)
        fx = torch.ones_like(box_w) * fx
        fy = torch.ones_like(box_w) * fx
        cx = torch.zeros_like(box_w)
        cy = torch.zeros_like(box_w)
        hand_crop_h = torch.ones_like(box_w) * hand_crop_h
        hand_crop_w = torch.ones_like(box_w) * hand_crop_w
        hand_cam = BatchCameraManager(
            fx=fx, fy=fy, cx=cx, cy=cy, img_h=box_h, img_w=box_w,
            in_ndc=True
        ).resize(hand_crop_h, hand_crop_w)
        return hand_cam

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

    @staticmethod
    def pad_and_crop(image: torch.Tensor,
                     box: torch.Tensor,
                     out_size: int) -> torch.Tensor:
        """ Pad 0's if box exceeds boundary.

        Args:
            image: torch.Tensor (..., H, W)
            box: torch.Tensor (4) xywh
            out_size: int

        Returns:
            img_crop: torch.Tensor (..., crop_h, crop_w) in [0, 1]
        """
        x, y, w, h = box.clone()  # or call item() to avoid pointer
        img_h, img_w = image.shape[-2:]
        pad_x = max(max(-x ,0), max(x+w-img_w, 0))
        pad_y = max(max(-y ,0), max(y+h-img_h, 0))
        transform = transforms.Compose(
            [transforms.Pad([pad_x, pad_y])])
        x += pad_x
        y += pad_y

        image_pad = transform(image)
        crop_tensor = F.resized_crop(
            image_pad.unsqueeze(0),
            int(y), int(x), int(h), int(w), size=[out_size, out_size],
            interpolation=transforms.InterpolationMode.NEAREST
        )[0]
        return crop_tensor

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
        image_patch = [None for _ in range(bsize)]
        obj_mask_patch = [None for _ in range(bsize)]
        for i in range(bsize):
            obj_mask_patch[i] = self.pad_and_crop(
                object_mask[i], obj_bbox_squared[i], self.rend_size)

            _image = transforms.ToTensor()(image[i])
            _image_patch = self.pad_and_crop(
                _image, obj_bbox_squared[i], self.rend_size)
            image_patch[i] = np.asarray(transforms.ToPILImage()(_image_patch))
        image_patch = np.stack(image_patch, 0)
        obj_mask_patch = torch.stack(obj_mask_patch, 0)

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
            square_bbox=obj_bbox_squared,
            mask=obj_mask_patch,
            K_global=self.global_cam.get_K(),

            rotations_init=rotations_init,
            translations_init=translations_init,
            base_rotation=base_rotation,
            base_translation=base_translation,
            num_initializations=num_initializations,
            num_iterations=num_iterations,
            debug=debug,
            sort_best=sort_best,
            viz=viz,
            lr=lr,
        )

        self._fit_model = model
        return self._fit_model


def find_optimal_pose(
    vertices,
    faces,
    mask,
    bbox,
    square_bbox,
    K_global,
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
    base_translation=None,
    viz=True):
    """
    Args:
        vertices: torch.Tensor (V, 3)
        faces: torch.Tensor (F, 3)
        mask: troch.Tensor (B, W, W)
            1 for fg, 0 for bg, -1 for ignored
        bbox: torch.Tensor (B, 4)
            XYWH bbox from original data source. E.g. epichoa
        square_bbox: torch.Tensor (B, 4)
            XYWH Enlarged and squared from `bbox`
            The box that matches `mask`
        K_global: torch.Tensor (B, 3, 3), represents the global camera that
            captures the original image.
        base_rotation: torch.Tensor (B, 3, 3)
        base_translation: torch.Tensor (B, 3)

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

    x, y, b, _ = torch.split(square_bbox, [1, 1, 1, 1], dim=1)  # x,y,b: (B,1)
    crop_boxes = torch.cat([x, y, x+b, y+b], dim=1)  # (B, 4)
    camintr_bbox = kcrop.get_K_crop_resize(
        torch.as_tensor(K_global), crop_boxes,
        [REND_SIZE]).to(device)
    # Equivalently: K.crop(square_box).resize(REND_SIZE)

    # Stuff to keep around
    best_losses = torch.tensor(np.inf)
    best_rots = None
    best_trans = None
    best_loss_single = torch.tensor(np.inf)
    best_rots_single = None
    best_trans_single = None
    loop = tqdm(total=num_iterations)
    K_global = K_global.to(device)
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
            [bsize, 3], dtype=rotations_init.dtype, device=device)
    else:
        base_translation = base_translation.clone()

    V_rotated = torch.matmul(
            (vertices @ base_rotation).unsqueeze_(1),
            rotations_init.unsqueeze(0))
    if translations_init is None:
        # Init B trans_init, each has N_init, then take mean of them
        translations_init = []
        for i in range(bsize):
            _translations_init = TCO_init_from_boxes_zup_autodepth(
                bbox[i], V_rotated[i], K_global[i])
            _translations_init -= base_translation[i] @ rotations_init
            translations_init.append(_translations_init)
    translations_init = sum(translations_init) / len(translations_init)

    if debug:
        mask_viz = mask[0]  # select 0-th mask for visualization
        # Debug shows initalized verts on image & mask
        trans_verts = translations_init + torch.matmul(vertices,
                                                       rotations_init)
        proj_verts = project.batch_proj2d(trans_verts,
                                          K_global.repeat(trans_verts.shape[0], 1,
                                                   1)).cpu()
        verts3d = trans_verts.cpu()
        flat_verts = proj_verts.contiguous().view(-1, 2)
        if viz:
            plt.clf()
            fig, axes = plt.subplots(1, 3)
            ax = axes[0]
            ax.scatter(flat_verts[:, 0], flat_verts[:, 1], s=1, alpha=0.2)
            ax = axes[1]
            for vert in proj_verts:
                ax.scatter(vert[:, 0], vert[:, 1], s=1, alpha=0.2)
            ax = axes[2]
            for vert in verts3d:
                ax.scatter(vert[:, 0], vert[:, 2], s=1, alpha=0.2)

            fig.savefig(os.path.join(viz_folder, "autotrans.png"))
            plt.close()

        proj_verts = project.batch_proj2d(
            trans_verts, camintr_bbox.repeat(trans_verts.shape[0], 1, 1)).cpu()
        flat_verts = proj_verts.contiguous().view(-1, 2)
        if viz:
            fig, axes = plt.subplots(1, 3)
            ax = axes[0]
            ax.imshow(mask_viz)
            ax.scatter(flat_verts[:, 0], flat_verts[:, 1], s=1, alpha=0.2)
            ax = axes[1]
            ax.imshow(mask_viz)
            for vert in proj_verts:
                ax.scatter(vert[:, 0], vert[:, 1], s=1, alpha=0.2)
            ax = axes[2]
            for vert in verts3d:
                ax.scatter(vert[:, 0], vert[:, 2], s=1, alpha=0.2)
            fig.savefig(os.path.join(viz_folder, "autotrans_roi.png"))
            plt.close()

    # Bring crop K to NC rendering space
    camintr_bbox[:, :2] = camintr_bbox[:, :2] / REND_SIZE

    model = PoseRenderer(
        ref_image=mask,
        vertices=vertices,
        faces=faces,
        rotation_init=matrix_to_rot6d(rotations_init),
        translation_init=translations_init,
        num_initializations=num_initializations,
        camera_K=camintr_bbox,
        base_rotation=base_rotation,
        base_translation=base_translation
    )
    # model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(num_iterations):
        optimizer.zero_grad()
        loss_dict, iou, sil = model()
        if debug and (step % viz_step == 0):
            debug_viz_folder = os.path.join(viz_folder, "poseoptim")
            os.makedirs(debug_viz_folder, exist_ok=True)
            imagify.viz_imgrow(
                sil, overlays=[mask_viz,]*len(sil), viz_nb=4,
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
