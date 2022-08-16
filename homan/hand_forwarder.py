import cv2
from typing import List
import neural_renderer as nr
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

from nnutils.handmocap import recover_pca_pose
from homan.homan_ManoModel import HomanManoModel
from homan.lossutils import iou_loss, rotation_loss_v1
from homan.utils.geometry import matrix_to_rot6d, rot6d_to_matrix
from homan.ho_utils import (
    compute_transformation_ortho, compute_transformation_persp)
from nnutils.hand_utils import ManopthWrapper

import roma
from libzhifan.geometry import (
    SimpleMesh, visualize_mesh, projection, CameraManager)
from libzhifan.numeric import check_shape

from libyana.conversions import npt
from libyana.metrics.iou import batch_mask_iou

from obj_pose.pose_optimizer import PoseOptimizer


"""
Batch dimension == Time dimension
"""


class HandForwarder(nn.Module):
    def __init__(self,
                 translations_hand,  # (B, 3)
                 rotations_hand,
                 verts_hand_og,
                 hand_sides,
                 mano_trans,
                 mano_rot,
                 mano_betas,
                 mano_pca_pose,
                 faces_hand,

                 camintr,
                 target_masks_hand,
                 cams_hand=None,
                 scale_hand=1.0,
                 optimize_ortho_cam=True,
                 hand_proj_mode="persp",
                 optimize_mano=True,
                 optimize_mano_beta=True,
                 image_size:int = 640,

                 ihoi_img_patch=None,
                 ):
        """
        Hands are received in batch of [h_1_t_1, h_2_t_1, ..., h_1_t_2]
        (h_{hand_index}_t_{time_step})

        Args:
            verts_hand_og: Transformed hands 
            target_masks_hand: used as _ref_mask_hand
                and to be used in self.renderer
            ihoi_img_patch: optional. Background of render_scene
        """
        super().__init__()
        self.ihoi_img_patch = ihoi_img_patch
        bsize = len(camintr)
        self.bsize = bsize

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
        if optimize_ortho_cam:
            self.cams_hand = nn.Parameter(cams_hand, requires_grad=True)
        else:
            self.register_buffer("cams_hand", cams_hand)
        self.hand_sides = hand_sides
        self.hand_nb = len(hand_sides)

        self.optimize_mano = optimize_mano
        if optimize_mano:
            self.mano_pca_pose = nn.Parameter(mano_pca_pose,
                                              requires_grad=True)
            self.mano_rot = nn.Parameter(mano_rot, requires_grad=True)
            self.mano_trans = nn.Parameter(mano_trans, requires_grad=True)
        else:
            self.register_buffer("mano_pca_pose", mano_pca_pose)
            self.register_buffer("mano_rot", mano_rot)
        if optimize_mano_beta:
            self.mano_betas = nn.Parameter(torch.zeros_like(mano_betas),
                                           requires_grad=True)
            self.register_buffer("scale_hand",
                                 torch.as_tensor([scale_hand]))
        else:
            self.register_buffer("mano_betas", torch.zeros_like(mano_betas))
            self.scale_hand = nn.Parameter(
                scale_hand * torch.ones(1).float(),
                requires_grad=True)
        self.register_buffer("verts_hand_og", verts_hand_og)
        self.register_buffer("int_scale_hand_mean",
                             torch.Tensor([1.0]).float().cuda())
        self.register_buffer("ref_mask_hand", (target_masks_hand > 0).float())
        self.register_buffer("keep_mask_hand",
                             (target_masks_hand >= 0).float())

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
        self.renderer = nr.renderer.Renderer(
            image_size=256,
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

        self._check_shape(bsize=self.bsize)

    def _check_shape(self, bsize):
        check_shape(self.faces_hand, (bsize, -1, 3))
        check_shape(self.camintr, (bsize, 3, 3))
        mask_shape = self.ref_mask_hand.shape
        check_shape(self.ref_mask_hand, mask_shape)
        check_shape(self.keep_mask_hand, mask_shape)
        check_shape(self.rotations_hand, (bsize, 3, 2))
        check_shape(self.translations_hand, (bsize, 1, 3))
        # ordinal loss
        check_shape(self.masks_human, (bsize, 1, -1, -1))

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
        if self.optimize_mano:
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
        else:
            verts_hand_og = self.verts_hand_og
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
    
    def render(self) -> torch.Tensor:
        """ returns: (B, W, W) betwenn [0, 1] """
        # Rendering happens in ROI
        rend = self.renderer(
            self.get_verts_hand(),
            self.faces_hand,
            K=self.camintr,
            mode="silhouettes")
        image = self.keep_mask_hand * rend
        return image

    def forward_sil(self, compute_iou=True, func='l2'):
        """ 
        Returns:
            loss: torch.Tensor (B,)
            iou: (B,)
        """
        # Rendering happens in ROI
        rend = self.renderer(
            self.get_verts_hand(),
            self.faces_hand,
            K=self.camintr,
            mode="silhouettes")
        image = self.keep_mask_hand * rend
        if func == 'l2':
            loss_sil = torch.sum(
                (image - self.ref_mask_hand)**2, dim=(1, 2)) # / self.keep_mask_hand.sum()
        elif func == 'iou':
            loss_sil = iou_loss(image, self.ref_mask_hand)
        
        loss_sil = loss_sil / self.bsize
        if compute_iou:
            ious = batch_mask_iou(image, self.ref_mask_hand)
            return loss_sil, ious
        return loss_sil

    def loss_pca_interpolation(self) -> torch.Tensor:
        """
        Prior: pose(t) = (pose(t+1) + pose(t-1)) / 2

        Returns: (B-2,)
        """
        target = (self.mano_pca_pose[2:] + self.mano_pca_pose[:-2]) / 2
        pred = self.mano_pca_pose[1:-1]
        loss = torch.mean((target - pred)**2, dim=(1))
        return loss

    def loss_rot_interpolation(self) -> torch.Tensor:
        device = self.rotations_hand.device
        rotmat = rot6d_to_matrix(self.rotations_hand)
        rot_mid = roma.rotmat_slerp(
            rotmat[2:], rotmat[:-2],
            torch.as_tensor([0.5], device=device))[0]
        loss = rotation_loss_v1(rot_mid, rotmat[1:-1])
        return loss

    def loss_transl_interpolation(self) -> torch.Tensor:
        """
        Returns: (B-2,)
        """
        interp = (self.translations_hand[2:] + self.translations_hand[:-2]) / 2
        pred = self.translations_hand[1:-1]
        loss = torch.sum((interp - pred)**2, dim=(1, 2))
        return loss
    
    def forward_hand(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def forward(self,
                loss_weights={
                    'sil': 1,
                    'pca': 1,
                    'rot': 1,
                    'transl': 1,
                }):
        l_sil = self.forward_sil(compute_iou=False, func='iou').sum()
        l_pca = self.loss_pca_interpolation().sum()
        l_rot = self.loss_rot_interpolation().sum()
        l_transl = self.loss_transl_interpolation().sum()
        losses = {
            'sil': l_sil,
            'pca': l_pca,
            'rot': l_rot,
            'transl': l_transl,
        }
        for k, l in losses.items():
            losses[k] = l * loss_weights[k]
        total_loss = sum([v for v in losses.values()])
        return total_loss, losses
    
    def get_meshes(self, idx, **mesh_kwargs) -> List[SimpleMesh]:
        hand_color = mesh_kwargs.pop('hand_color', 'light_blue')
        with torch.no_grad():
            verts_hand = self.get_verts_hand(**mesh_kwargs)[idx]
            mhand = SimpleMesh(
                verts_hand, self.faces_hand[idx], tex_color=hand_color)
        return mhand

    def to_scene(self, idx=-1, show_axis=False, viewpoint='nr', **mesh_kwargs):
        """ Returns a trimesh.Scene """
        if idx >= 0:
            mhand = self.get_meshes(idx=idx, **mesh_kwargs)
            return visualize_mesh(mhand,
                                  show_axis=show_axis,
                                  viewpoint=viewpoint)

        """ Render all """
        hand_color = mesh_kwargs.pop('hand_color', 'light_blue')
        with torch.no_grad():
            verts_hand = self.get_verts_hand(**mesh_kwargs)

        meshes = []
        disp = 0.15  # displacement
        for t in range(self.bsize):
            mhand = SimpleMesh(
                verts_hand[t], self.faces_hand[t], tex_color=hand_color)
            mhand.apply_translation_([t * disp, 0, 0])
            meshes.append(mhand)
        return visualize_mesh(meshes, show_axis=show_axis)

    def render_summary(self, idx) -> np.ndarray:
        a1 = np.uint8(self.ihoi_img_patch[idx])
        mask_hand = self.ref_mask_hand[idx].cpu().numpy().squeeze()
        all_mask = np.zeros_like(a1, dtype=np.float32)
        all_mask = np.where(
            mask_hand[...,None], (0, 0, 0.8), all_mask)
        all_mask = np.uint8(255*all_mask)
        a2 = cv2.addWeighted(a1, 0.9, all_mask, 0.5, 1.0)
        a3 = np.uint8(self.render_scene(idx=idx)*255)
        b = np.uint8(255*self.render_triview(idx=idx))
        a = np.hstack([a3, a2, a1])
        return np.vstack([a,
                          b])

    def render_scene(self, idx, with_hand=True, **mesh_kwargs) -> np.ndarray:
        """ returns: (H, W, 3) """
        if not with_hand:
            return self.ihoi_img_patch[idx]
        mhand = self.get_meshes(idx=idx, **mesh_kwargs)
        img = projection.perspective_projection_by_camera(
            [mhand],
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
        mhand = self.get_meshes(idx=idx, **mesh_kwargs)
        front = projection.project_standardized(
            [mhand],
            direction='+z',
            image_size=image_size,
            method=dict(
                name='pytorch3d',
                coor_sys='nr',
                in_ndc=False
            )
        )
        left = projection.project_standardized(
            [mhand],
            direction='+x',
            image_size=image_size,
            method=dict(
                name='pytorch3d',
                coor_sys='nr',
                in_ndc=False
            )
        )
        back = projection.project_standardized(
            [mhand],
            direction='-z',
            image_size=image_size,
            method=dict(
                name='pytorch3d',
                coor_sys='nr',
                in_ndc=False
            )
        )
        return np.hstack([front, left, back])

    def render_grid(self, with_hand=True):
        l = self.bsize
        num_cols = 5
        num_rows = (l + num_cols - 1) // num_cols
        fig, axes = plt.subplots(
            nrows=num_rows, ncols=num_cols,
            sharex=True, sharey=True, figsize=(20, 20))
        for cam_idx, ax in enumerate(axes.flat, start=0):
            img = self.render_scene(idx=cam_idx, with_hand=with_hand)
            ax.imshow(img)
            ax.set_axis_off()
            if cam_idx == l-1:
                break
        plt.tight_layout()
        return fig


def init_hand_forwarder(one_hands, 
                        images, 
                        side: str, 
                        obj_bboxes,
                        hand_masks) -> HandForwarder:
    pose_machine = PoseOptimizer(
        one_hands, obj_loader=None, side=side)

    _, image_patch, hand_mask_patch = pose_machine.finalize_without_fit(
        images, obj_bboxes, hand_masks)

    bsize = len(pose_machine)
    mano_pca_pose = recover_pca_pose(pose_machine.pred_hand_pose, side=side)
    mano_rot = torch.zeros([bsize, 3], device=mano_pca_pose.device)
    mano_trans = torch.zeros([bsize, 3], device=mano_pca_pose.device)
    camintr = pose_machine.ihoi_cam.to_nr(return_mat=True)  # could be pose_machine.ihoi_cam

    hand_kwargs = dict(
        translations_hand = pose_machine.hand_translation,
        rotations_hand = pose_machine.hand_rotation,
        verts_hand_og = pose_machine.hand_verts,
        hand_sides = [side],
        mano_trans = mano_trans,
        mano_rot = mano_rot,
        mano_betas = pose_machine.pred_hand_betas,
        mano_pca_pose = mano_pca_pose,
        faces_hand = pose_machine.hand_faces,

        scale_hand = 1.0,

        camintr = camintr,
        target_masks_hand = hand_mask_patch,

        image_size = pose_machine.rend_size,
        ihoi_img_patch=image_patch,
        )

    for k, v in hand_kwargs.items():
        if hasattr(v, 'device'):
            hand_kwargs[k] = v.to('cuda')
    homan = HandForwarder(**hand_kwargs).cuda()
    return homan
