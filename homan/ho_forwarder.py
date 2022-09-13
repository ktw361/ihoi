import cv2
from typing import List
import neural_renderer as nr
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

from homan import lossutils
from homan.losses import Losses
from homan.homan_ManoModel import HomanManoModel
from homan.utils.geometry import matrix_to_rot6d, rot6d_to_matrix
from homan.ho_utils import (
    compute_transformation_ortho, compute_transformation_persp)
from homan.lossutils import rotation_loss_v1
import roma

from libzhifan.geometry import (
    SimpleMesh, visualize_mesh, projection, CameraManager)

from libyana.conversions import npt
from libzhifan.numeric import check_shape


""" A modified version of homan_core.py

Ps 1.
Translation is inheritly related to scale.
if Y = RX + T, then sY = sRX + sT.

Ps 2.
Batch dimension == Time dimension
"""

class HOForwarder(nn.Module):
    def __init__(self,
                 translations_object,  # (1, 3)
                 rotations_object,
                 verts_object_og,
                 faces_object,

                 translations_hand,  # (B, 3)
                 rotations_hand,
                 hand_sides,
                 mano_trans,
                 mano_rot,
                 mano_betas,
                 mano_pca_pose,
                 faces_hand,

                 camintr,
                 target_masks_object,
                 target_masks_hand,
                 class_name='default',
                 scale_hand=1.0,
                 scale_object=1.0,
                 hand_proj_mode="persp",
                 inter_type="centroid",
                 image_size:int = 640,

                 ihoi_img_patch=None,
                 ):
        """
        Hands are received in batch of [h_1_t_1, h_2_t_1, ..., h_1_t_2]
        (h_{hand_index}_t_{time_step})

        Args:
            target_masks_object: used as ref_mask_obj
            target_masks_hand: used as _ref_mask_hand
            camintr_rois_object: from model.K.detach(),
                and to be used in self.renderer
            ihoi_img_patch: optional. Background of render_scene
        """
        super().__init__()
        self.ihoi_img_patch = ihoi_img_patch
        bsize = len(camintr)
        self.bsize = bsize

        """ Initialize object pamaters """
        if rotations_object.shape[-1] == 3:
            rotations_object6d = matrix_to_rot6d(rotations_object)
        else:
            rotations_object6d = rotations_object
        self.rotations_object = nn.Parameter(
            rotations_object6d.detach().clone(), requires_grad=True)
        self.translations_object = nn.Parameter(
            translations_object.detach().clone(),
            requires_grad=True)
        self.register_buffer("verts_object_og", verts_object_og)
        # Zhifan: Note we modify translation to be a function of scale:
        # T(s) = s * T_init
        self.scale_object = nn.Parameter(
            torch.as_tensor([scale_object]),
            requires_grad=True,
        )
        self.register_buffer("int_scale_object_mean", torch.ones(1).float())


        """ Inititalize person parameters """
        # TODO(zhifan): in HOMAN num_pca = 16
        hand_side = hand_sides[0]
        self.mano_model = HomanManoModel("externals/mano", side=hand_side, pca_comps=45)
        self.hand_proj_mode = hand_proj_mode
        translation_init = translations_hand.detach().clone()
        self.translations_hand = nn.Parameter(translation_init,
                                              requires_grad=True)
        rotations_hand = rotations_hand.detach().clone()
        if rotations_hand.shape[-1] == 3:
            rotations_hand = matrix_to_rot6d(rotations_hand)
        self.rotations_hand = nn.Parameter(rotations_hand, requires_grad=True)
        self.hand_sides = hand_sides
        self.hand_nb = len(hand_sides)

        self.mano_pca_pose = nn.Parameter(mano_pca_pose, requires_grad=True)
        self.mano_rot = nn.Parameter(mano_rot, requires_grad=True)
        self.mano_trans = nn.Parameter(mano_trans, requires_grad=True)
        self.mano_betas = nn.Parameter(torch.zeros_like(mano_betas),
                                        requires_grad=True)
        self.scale_hand = nn.Parameter(
            scale_hand * torch.ones(1).float(),
            requires_grad=True)

        self.register_buffer("int_scale_hand_mean",
                             torch.Tensor([1.0]).float().cuda())
        self.register_buffer("ref_mask_object",
                             (target_masks_object > 0).float())
        self.register_buffer("keep_mask_object",
                             (target_masks_object >= 0).float())
        self.register_buffer("ref_mask_hand", (target_masks_hand > 0).float())
        self.register_buffer("keep_mask_hand",
                             (target_masks_hand >= 0).float())

        self.register_buffer("faces_object", faces_object.repeat(bsize, 1, 1))
        self.register_buffer(
            "textures_object",
            torch.ones(bsize, faces_object.shape[1], 1, 1, 1, 3))
        self.register_buffer(
            "textures_hand",
            torch.ones(bsize, faces_hand.shape[1], 1, 1, 1, 3))
        self.register_buffer("faces_hand", faces_hand.repeat(bsize, 1, 1))
        self.cuda()

        # Setup renderer
        if camintr is None:
            raise ValueError("Not allowed")
        else:
            camintr = npt.tensorify(camintr)
            if camintr.dim() == 2:
                camintr = camintr.unsqueeze(0)
            camintr = camintr.cuda().float()
        self.register_buffer("camintr", camintr)
        self.image_size = image_size
        self.renderer = nr.renderer.Renderer(image_size=self.image_size,
                                             K=camintr.clone(),
                                             R=torch.eye(3, device='cuda')[None],
                                             t=torch.zeros([1, 3], device='cuda'),
                                             orig_size=1)
        self.renderer.light_direction = [1, 0.5, 1]
        self.renderer.light_intensity_direction = 0.3
        self.renderer.light_intensity_ambient = 0.5
        self.renderer.background_color = [1.0, 1.0, 1.0]

        mask_h, mask_w = target_masks_hand.shape[-2:]
        # Following two are used by ordinal loss
        self.register_buffer(
            "masks_human", 
            target_masks_hand.view(self.bsize, 1, mask_h, mask_w).bool())
        self.register_buffer(
            "masks_object", 
            target_masks_object.view(self.bsize, 1, mask_h, mask_w).bool())

        self.losses = Losses(
            renderer=self.renderer,
            ref_mask_object=self.ref_mask_object,
            keep_mask_object=self.keep_mask_object,
            ref_mask_hand=self.ref_mask_hand,
            keep_mask_hand=self.keep_mask_hand,
            camintr=self.camintr,
            class_name=class_name,
            hand_nb=self.hand_nb,
            inter_type=inter_type,
        )
        self._check_shape(bsize=self.bsize)

    def _check_shape(self, bsize):
        check_shape(self.faces_hand, (bsize, -1, 3))
        check_shape(self.faces_object, (bsize, -1, 3))
        check_shape(self.camintr, (bsize, 3, 3))
        check_shape(self.ref_mask_object, (bsize, -1, -1))
        mask_shape = self.ref_mask_object.shape
        check_shape(self.keep_mask_object, mask_shape)
        check_shape(self.ref_mask_hand, mask_shape)
        check_shape(self.keep_mask_hand, mask_shape)
        check_shape(self.rotations_hand, (bsize, 3, 2))
        check_shape(self.translations_hand, (bsize, 1, 3))
        check_shape(self.rotations_object, (1, 3, 2))
        check_shape(self.translations_object, (1, 1, 3))
        # ordinal loss
        check_shape(self.masks_object, (bsize, 1, -1, -1))
        check_shape(self.masks_human, (bsize, 1, -1, -1))

    def get_verts_object(self,
                         translations_o2h=None,
                         scale_object=None,
                         **kwargs) -> torch.Tensor:
        """
            V_out = (V_model x R_o2h + T_o2h) x R_hand + T_hand
                  = V x (R_o2h x R_hand) + (T_o2h x R_hand + T_hand)
        where self.rotations/translations_object is R/T_o2h from object to hand

        Returns:
            verts_object: (B, V, 3)
        """
        rotations_o2h = rot6d_to_matrix(self.rotations_object)
        translations_o2h = self.translations_object \
            if translations_o2h is None else translations_o2h
        intrinsic_scales = self.scale_object if scale_object is None else scale_object
        """ Compound T_o2c (T_obj w.r.t camera) = T_h2c x To2h_ """
        R_hand = rot6d_to_matrix(self.rotations_hand)
        T_hand = self.translations_hand
        rot_o2c = rotations_o2h @ R_hand
        transl_o2c = translations_o2h @ R_hand + T_hand
        obj_verts = compute_transformation_persp(
            meshes=self.verts_object_og,
            translations=transl_o2c,
            rotations=rot_o2c,
            intrinsic_scales=intrinsic_scales
        )
        return obj_verts

    def get_joints_hand(self):
        all_hand_joints = []
        for hand_idx, side in enumerate(self.hand_sides):
            mano_pca_pose = self.mano_pca_pose[hand_idx::self.hand_nb]
            mano_rot = self.mano_rot[hand_idx::self.hand_nb]
            mano_res = self.mano_model.forward_pca(
                mano_pca_pose,
                rot=mano_rot,
                betas=self.mano_betas[hand_idx::self.hand_nb],
                side=side)
            joints = mano_res["joints"]
            verts = mano_res["verts"]
            # Add finger tips and reorder
            tips = verts[:, [745, 317, 444, 556, 673]]
            full_joints = torch.cat([joints, tips], 1)
            full_joints = full_joints[:, [
                0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7,
                8, 9, 20
            ]]
            all_hand_joints.append(full_joints)

        all_hand_joints = torch.stack(all_hand_joints).transpose(
            0, 1).contiguous().view(-1, 21, 3)
        joints_hand_og = all_hand_joints + self.mano_trans.unsqueeze(1)
        rotations_hand = rot6d_to_matrix(self.rotations_hand)
        return compute_transformation_persp(
            meshes=joints_hand_og,
            translations=self.translations_hand,
            rotations=rotations_hand,
            intrinsic_scales=self.scale_hand,
        )

    def get_verts_hand(self, detach_scale=False, **kwargs):
        all_hand_verts = []
        for hand_idx, side in enumerate(self.hand_sides):
            mano_pca_pose = self.mano_pca_pose[hand_idx::self.hand_nb]
            mano_rot = self.mano_rot[hand_idx::self.hand_nb]
            mano_res = self.mano_model.forward_pca(
                mano_pca_pose,
                rot=mano_rot,
                betas=self.mano_betas[hand_idx::self.hand_nb],
                side=side)
            vertices = mano_res["verts"]
            all_hand_verts.append(vertices)
        all_hand_verts = torch.stack(all_hand_verts).transpose(
            0, 1).contiguous().view(-1, 778, 3)
        verts_hand_og = all_hand_verts + self.mano_trans.unsqueeze(1)
        if detach_scale:
            scale = self.scale_hand.detach()
        else:
            scale = self.scale_hand
        rotations_hand = rot6d_to_matrix(self.rotations_hand)
        if self.hand_proj_mode == "ortho":
            raise NotImplementedError
            return compute_transformation_ortho(
                meshes=verts_hand_og,
                cams=self.cams_hand,
                intrinsic_scales=scale,
                K=self.renderer.K,
                img=self.masks_human,
            )
        elif self.hand_proj_mode == "persp":
            return compute_transformation_persp(
                meshes=verts_hand_og,
                translations=self.translations_hand,
                rotations=rotations_hand,
                intrinsic_scales=scale,
            )
        else:
            raise ValueError(
                f"Expected hand_proj_mode {self.hand_proj_mode} to be in [ortho|persp]"
            )

    def checkpoint(self, session=0):
        torch.save(self.state_dict(), f'/tmp/h{session}.pth')
    
    def resume(self, session=0):
        self.load_state_dict(torch.load(f'/tmp/h{session}.pth'), strict=True)

    def compute_ordinal_depth_loss(self, 
                                   verts_hand,
                                   verts_object,
                                   return_aux=False, 
                                   **mesh_kwargs):
        silhouettes = []
        depths = []

        _, depths_o, silhouettes_o = self.renderer.render(
            verts_object, self.faces_object, self.textures_object)
        silhouettes_o = (silhouettes_o == 1).bool()
        silhouettes.append(silhouettes_o)
        depths.append(depths_o)

        hand_masks = []
        for hand_idx, faces, textures in zip(range(self.hand_nb),
                                             self.faces_hand,
                                             self.textures_hand):
            hand_verts = verts_hand[hand_idx::self.hand_nb]
            repeat_nb = hand_verts.shape[0]
            faces_hand = faces.unsqueeze(0).repeat(repeat_nb, 1, 1)
            textures_hand = textures.unsqueeze(0).repeat(
                repeat_nb, 1, 1, 1, 1, 1)
            _, depths_p, silhouettes_p = self.renderer.render(
                hand_verts, faces_hand, textures_hand)
            silhouettes_p = (silhouettes_p == 1).bool()
            # imagify.viz_imgrow(cols, "tmp_hands.png")
            silhouettes.append(silhouettes_p)
            depths.append(depths_p)
            hand_masks.append(self.masks_human[hand_idx::self.hand_nb])

        all_masks = [self.masks_object] + [
            self.masks_human[hand_idx::self.hand_nb]
            for hand_idx in range(self.hand_nb)
        ]
        masks = torch.cat(all_masks, 1)
        loss_dict = lossutils.compute_ordinal_depth_loss(
            masks, silhouettes, depths)
        if return_aux:
            return masks, silhouettes, depths, loss_dict
        else:
            return loss_dict

    def loss_sil_hand(self):
        verts_hand = self.get_verts_hand()
        loss_dict, ious = self.losses.compute_sil_loss_hand(
            verts_hand, self.faces_hand, 
            compute_iou=False, func='iou')
        loss_sil = loss_dict['sil_hand']
        return loss_sil

    def loss_pose_interpolation(self) -> torch.Tensor:
        """
        Interpolation Prior: pose(t) = (pose(t+1) + pose(t-1)) / 2

        Returns: (B-2,)
        """
        target = (self.mano_pca_pose[2:] + self.mano_pca_pose[:-2]) / 2
        pred = self.mano_pca_pose[1:-1]
        loss = torch.mean((target - pred)**2, dim=(1))
        return loss

    def loss_hand_rot(self) -> torch.Tensor:
        """ Interpolation loss for hand rotation """
        device = self.rotations_hand.device
        rotmat = rot6d_to_matrix(self.rotations_hand)
        rot_mid = roma.rotmat_slerp(
            rotmat[2:], rotmat[:-2],
            torch.as_tensor([0.5], device=device))[0]
        loss = rotation_loss_v1(rot_mid, rotmat[1:-1])
        return loss

    def loss_hand_transl(self) -> torch.Tensor:
        """
        Returns: (B-2,)
        """
        interp = (self.translations_hand[2:] + self.translations_hand[:-2]) / 2
        pred = self.translations_hand[1:-1]
        loss = torch.sum((interp - pred)**2, dim=(1, 2))
        return loss
    
    def forward(self, loss_weights=None, **mesh_kwargs):
        """
        If a loss weight is zero, that loss isn't computed (to avoid unnecessary
        compute).
        """
        loss_dict = {}
        metric_dict = {}
        verts_object = self.get_verts_object(**mesh_kwargs)
        # verts_hand_det has MANO mesh detached, which allows to backpropagate
        # coarse interaction loss simply in translation
        verts_hand = self.get_verts_hand(**mesh_kwargs)
        verts_hand_det_scale = self.get_verts_hand(detach_scale=True)
        if loss_weights is None or loss_weights["pca"] > 0:
            loss_pca = lossutils.compute_pca_loss(self.mano_pca_pose)
            loss_dict.update(loss_pca)
        if loss_weights is None or loss_weights["collision"] > 0:
            # Pushes hand out of object, gradient not flowing through object !
            loss_coll = lossutils.compute_collision_loss(
                verts_hand=verts_hand_det_scale,
                verts_object=verts_object.detach(),
                faces_object=self.faces_object,
                faces_hand=self.faces_hand)
            loss_dict.update(loss_coll)

        if loss_weights is None or loss_weights["contact"] > 0:
            loss_contact = lossutils.compute_contact_loss(
                verts_hand_b=verts_hand_det_scale,
                verts_object_b=verts_object,
                faces_object=self.faces_object)
            loss_dict.update(loss_contact)
        if loss_weights is None or loss_weights["sil_obj"] > 0:
            sil_loss_dict, sil_metric_dict = self.losses.compute_sil_loss_object(
                verts=verts_object, faces=self.faces_object)
            loss_dict.update(sil_loss_dict)
            metric_dict.update(sil_metric_dict)

        if loss_weights is None or loss_weights['sil_hand'] > 0:
            loss_dict.update(
                self.losses.compute_sil_loss_hand(verts=verts_hand,
                                                  faces=self.faces_hand))
        if loss_weights is None or loss_weights["inter"] > 0:
            # Interaction acts only on hand !
            inter_verts_object = verts_object.unsqueeze(1)
            verts_hand_b=verts_hand.view(-1, self.hand_nb, 778, 3)
            inter_loss_dict, inter_metric_dict = \
                self.losses.compute_interaction_loss(
                verts_hand_b=verts_hand_b,
                verts_object_b=inter_verts_object)
            loss_dict.update(inter_loss_dict)
            metric_dict.update(inter_metric_dict)

        if loss_weights is None or loss_weights["scale_obj"] > 0:
            loss_dict[
                "loss_scale_obj"] = lossutils.compute_intrinsic_scale_prior(
                    intrinsic_scales=self.scale_object,
                    intrinsic_mean=self.int_scale_object_mean,
                )
        if loss_weights is None or loss_weights["scale_hand"] > 0:
            loss_dict[
                "loss_scale_hand"] = lossutils.compute_intrinsic_scale_prior(
                    intrinsic_scales=self.scale_hand,
                    intrinsic_mean=self.int_scale_hand_mean,
                )
        if loss_weights is None or loss_weights["depth"] > 0:
            loss_dict.update(self.compute_ordinal_depth_loss(
                verts_hand, verts_object, **mesh_kwargs))
        return loss_dict, metric_dict

    def get_meshes(self, idx, **mesh_kwargs) -> List[SimpleMesh]:
        hand_color = mesh_kwargs.pop('hand_color', 'light_blue')
        obj_color = mesh_kwargs.pop('obj_color', 'yellow')
        with torch.no_grad():
            verts_obj = self.get_verts_object(**mesh_kwargs)[idx]
            verts_hand = self.get_verts_hand(**mesh_kwargs)[idx]
            mhand = SimpleMesh(
                verts_hand, self.faces_hand[idx], tex_color=hand_color)
            mobj = SimpleMesh(
                verts_obj, self.faces_object[idx], tex_color=obj_color)
        return mhand, mobj

    def to_scene(self, idx=-1, show_axis=False, viewpoint='nr', **mesh_kwargs):
        """ Returns a trimesh.Scene """
        if idx >= 0:
            mhand, mobj = self.get_meshes(idx=idx, **mesh_kwargs)
            return visualize_mesh([mhand, mobj],
                                show_axis=show_axis,
                                viewpoint=viewpoint)

        """ Render all """
        hand_color = mesh_kwargs.pop('hand_color', 'light_blue')
        obj_color = mesh_kwargs.pop('obj_color', 'yellow')
        with torch.no_grad():
            verts_obj = self.get_verts_object(**mesh_kwargs)
            verts_hand = self.get_verts_hand(**mesh_kwargs)

        meshes = []
        disp = 0.15  # displacement
        for t in range(self.bsize):
            mhand = SimpleMesh(
                verts_hand[t], self.faces_hand[t], tex_color=hand_color)
            mobj = SimpleMesh(
                verts_obj[t], self.faces_object[t], tex_color=obj_color)
            mhand.apply_translation_([t * disp, 0, 0])
            mobj.apply_translation_([t * disp, 0, 0])
            meshes.append(mhand)
            meshes.append(mobj)
        return visualize_mesh(meshes, show_axis=show_axis)

    def render_summary(self, idx) -> np.ndarray:
        a1 = np.uint8(self.ihoi_img_patch[idx])
        mask_obj = self.ref_mask_object[idx].cpu().numpy().squeeze()
        mask_hand = self.ref_mask_hand[idx].cpu().numpy().squeeze()
        all_mask = np.zeros_like(a1, dtype=np.float32)
        all_mask = np.where(
            mask_obj[...,None], (0.5, 0.5, 0), all_mask)
        all_mask = np.where(
            mask_hand[...,None], (0, 0, 0.8), all_mask)
        all_mask = np.uint8(255*all_mask)
        a2 = cv2.addWeighted(a1, 0.9, all_mask, 0.5, 1.0)
        a3 = np.uint8(self.render_scene(idx=idx)*255)
        b = np.uint8(255*self.render_triview(idx=idx))
        a = np.hstack([a3, a2, a1])
        return np.vstack([a,
                          b])

    def render_scene(self, idx, **mesh_kwargs) -> np.ndarray:
        """ returns: (H, W, 3) """
        mhand, mobj = self.get_meshes(idx=idx, **mesh_kwargs)
        img = projection.perspective_projection_by_camera(
            [mhand, mobj],
            CameraManager.from_nr(
                self.camintr.detach().cpu().numpy()[idx], self.image_size),
            method=dict(
                name='pytorch3d',
                coor_sys='nr',
                in_ndc=False
            ),
            image=self.ihoi_img_patch[idx],
        )
        return img

    def render_triview(self, idx, **mesh_kwargs) -> np.ndarray:
        """
        Returns:
            (H, W, 3)
        """
        image_size = 256
        mhand, mobj = self.get_meshes(idx=idx, **mesh_kwargs)
        front = projection.project_standardized(
            [mhand, mobj],
            direction='+z',
            image_size=image_size,
            method=dict(
                name='pytorch3d',
                coor_sys='nr',
                in_ndc=False
            )
        )
        left = projection.project_standardized(
            [mhand, mobj],
            direction='+x',
            image_size=image_size,
            method=dict(
                name='pytorch3d',
                coor_sys='nr',
                in_ndc=False
            )
        )
        back = projection.project_standardized(
            [mhand, mobj],
            direction='-z',
            image_size=image_size,
            method=dict(
                name='pytorch3d',
                coor_sys='nr',
                in_ndc=False
            )
        )
        return np.hstack([front, left, back])

    def render_grid(self):
        l = self.bsize
        num_cols = 5
        num_rows = (l + num_cols - 1) // num_cols
        fig, axes = plt.subplots(
            nrows=num_rows, ncols=num_cols,
            sharex=True, sharey=True, figsize=(20, 20))
        for cam_idx, ax in enumerate(axes.flat, start=0):
            img = self.render_scene(idx=cam_idx)
            ax.imshow(img)
            ax.set_axis_off()
            if cam_idx == l-1:
                break
        plt.tight_layout()
        return fig
