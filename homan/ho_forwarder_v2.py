import torch
import torch.nn as nn
import neural_renderer as nr
from scipy.ndimage.morphology import distance_transform_edt

from nnutils.handmocap import get_hand_faces
from homan.homan_ManoModel import HomanManoModel
from homan.utils.geometry import matrix_to_rot6d, rot6d_to_matrix
from homan.ho_utils import (
    compute_transformation_ortho, compute_transformation_persp)

from homan.lossutils import iou_loss, rotation_loss_v1
from homan.lossutils import rotation_loss_v1
import roma

from libzhifan.numeric import check_shape
from libzhifan.geometry import BatchCameraManager
from libyana.metrics.iou import batch_mask_iou


class HOForwarderV2(nn.Module):
    
    def __init__(self,
                 camintr: BatchCameraManager):
        """
        Args:
            camintr: (B, 3, 3). 
                Ihoi bounding box camera.
        """
        super().__init__()
        bsize = len(camintr)
        self.bsize = bsize
        self.mask_size = 256
        self.register_buffer("camintr", camintr)
        
        """ Set-up silhouettes renderer """
        self.renderer = nr.renderer.Renderer(
            image_size=self.mask_size,
            K=camintr.clone(),
            R=torch.eye(3, device='cuda')[None],
            t=torch.zeros([1, 3], device='cuda'),
            orig_size=1)
        self.renderer.light_direction = [1, 0.5, 1]
        self.renderer.light_intensity_direction = 0.3
        self.renderer.light_intensity_ambient = 0.5
        self.renderer.background_color = [1.0, 1.0, 1.0]
        # TODO: ordinal loss renderer image_size

    """ Common functions """

    def checkpoint(self, session=0):
        torch.save(self.state_dict(), f'/tmp/h{session}.pth')
    
    def resume(self, session=0):
        self.load_state_dict(torch.load(f'/tmp/h{session}.pth'), strict=True)

    """ Hand functions """

    def set_hand_params(self,
                        rotations_hand,
                        translations_hand,
                        hand_side: str,
                        mano_pca_pose,
                        mano_betas,
                        mano_trans=None,
                        mano_rot=None,
                        scale_hand=1.0):
        """ Inititalize person parameters """
        self.hand_sides = [hand_side]
        self.hand_nb = 1
        if mano_trans is None:
            mano_trans = torch.zeros([self.bsize, 3], device=mano_pca_pose.device)
        if mano_rot is None:
            mano_rot = torch.zeros([self.bsize, 3], device=mano_pca_pose.device)

        self.mano_model = HomanManoModel("externals/mano", side=hand_side, pca_comps=45)  # Note(zhifan): in HOMAN num_pca = 16
        translation_init = translations_hand.detach().clone()
        self.translations_hand = nn.Parameter(translation_init,
                                              requires_grad=True)
        rotations_hand = rotations_hand.detach().clone()
        if rotations_hand.shape[-1] == 3:
            rotations_hand = matrix_to_rot6d(rotations_hand)
        self.rotations_hand = nn.Parameter(rotations_hand, requires_grad=True)

        self.mano_pca_pose = nn.Parameter(mano_pca_pose, requires_grad=True)
        self.mano_rot = nn.Parameter(mano_rot, requires_grad=True)
        self.mano_trans = nn.Parameter(mano_trans, requires_grad=True)
        self.mano_betas = nn.Parameter(torch.zeros_like(mano_betas),
                                        requires_grad=True)
        self.scale_hand = nn.Parameter(
            scale_hand * torch.ones(1).float(),
            requires_grad=True)

        faces_hand = get_hand_faces(hand_side)
        self.register_buffer(
            "textures_hand",
            torch.ones(self.bsize, faces_hand.shape[1], 1, 1, 1, 3))
        self.register_buffer("faces_hand", faces_hand.repeat(self.bsize, 1, 1))
        self.cuda()

    def set_hand_target(self, target_masks_hand):
        self.register_buffer("ref_mask_hand", (target_masks_hand > 0).float())
        self.register_buffer("keep_mask_hand",
                             (target_masks_hand >= 0).float())
        mask_h, mask_w = target_masks_hand.shape[-2:]
        self.register_buffer(
            "masks_human", 
            target_masks_hand.view(self.bsize, 1, mask_h, mask_w).bool())
        self.cuda()
        self._check_shape_hand(self.bsize)

    def _check_shape_hand(self, bsize):
        check_shape(self.faces_hand, (bsize, -1, 3))
        check_shape(self.camintr, (bsize, 3, 3))
        check_shape(self.ref_mask_hand, (bsize, self.mask_size, self.mask_size))
        mask_shape = self.ref_mask_hand.shape
        check_shape(self.keep_mask_hand, mask_shape)
        check_shape(self.rotations_hand, (bsize, 3, 2))
        check_shape(self.translations_hand, (bsize, 1, 3))
        # ordinal loss
        check_shape(self.masks_human, (bsize, 1, self.mask_size, self.mask_size))
    
    @property
    def rot_mat_hand(self) -> torch.Tensor:
        return rot6d_to_matrix(self.rotations_hand)

    def get_verts_hand(self, detach_scale=False) -> torch.Tensor:
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
        rotations_hand = self.rot_mat_hand

        hand_proj_mode = 'persp'
        if hand_proj_mode == "ortho":
            raise NotImplementedError
            return compute_transformation_ortho(
                meshes=verts_hand_og,
                cams=self.cams_hand,
                intrinsic_scales=scale,
                K=self.renderer.K,
                img=self.masks_human,
            )
        elif hand_proj_mode == "persp":
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

    """ Object functions """

    def set_obj_params(self,
                       translations_object,
                       rotations_object,
                       verts_object_og,
                       faces_object,
                       scale_object=1.0):
        """ Initialize object pamaters

        Args:
            obj_trans: (N, 3, 3)
        """
        self.num_obj_init = len(rotations_object)
        if rotations_object.shape[-1] == 3:
            rotations_object6d = matrix_to_rot6d(rotations_object)
        else:
            rotations_object6d = rotations_object
        self.rotations_object = nn.Parameter(
            rotations_object6d.detach().clone(), requires_grad=True)
        self.translations_object = nn.Parameter(
            translations_object.detach().clone(),
            requires_grad=True)
        """ Translation is also a function of scale T(s) = s * T_init """
        self.scale_object = nn.Parameter(
            torch.as_tensor([scale_object]),
            requires_grad=True,
        )
        self.register_buffer("verts_object_og", verts_object_og)
        self.register_buffer(
            "faces_object", 
            faces_object.repeat(self.bsize * self.num_obj_init, 1, 1))
        self.register_buffer(
            "textures_object",
            torch.ones(self.bsize * self.num_obj_init, 
                       faces_object.shape[1], 1, 1, 1, 3))
    
    def set_obj_target(self, target_masks_object: torch.Tensor):
        """
        Args:
            target_masks_object: (B, W, W)
        """
        target_masks_object = target_masks_object
        self.register_buffer("ref_mask_object",
                             (target_masks_object > 0).float())
        self.register_buffer("keep_mask_object",
                             (target_masks_object >= 0).float())
        # edge loss is not used
        self.cuda()
        self._check_shape_object(self.num_obj_init)

    def _check_shape_object(self, num_init):
        check_shape(self.verts_object_og, (-1, 3))
        check_shape(self.faces_object, (self.bsize * num_init, -1, 3))
        check_shape(self.ref_mask_object,  (self.bsize, self.mask_size, self.mask_size))
        mask_shape = self.ref_mask_object.shape
        check_shape(self.keep_mask_object, mask_shape)
        check_shape(self.rotations_object, (num_init, 3, 2))
        check_shape(self.translations_object, (num_init, 1, 3))
        # ordinal loss TODO
        # check_shape(self.masks_object, (num_init, 1, self.mask_size, self.mask_size))
    
    @property
    def rot_mat_obj(self) -> torch.Tensor:
        return rot6d_to_matrix(self.rotations_object)

    def get_verts_object(self) -> torch.Tensor:
        """
            V_out = (V_model x R_o2h + T_o2h) x R_hand + T_hand
                  = V x (R_o2h x R_hand) + (T_o2h x R_hand + T_hand)
        where self.rotations/translations_object is R/T_o2h from object to hand

        Returns:
            verts_object: (B, N, V, 3)
        """
        R_o2h = self.rot_mat_obj  # (N, 3, 3)
        T_o2h = self.translations_object  # (N, 1, 3)
        scale = self.scale_object
        """ Compound T_o2c (T_obj w.r.t camera) = T_h2c x To2h_ """
        R_hand = self.rot_mat_hand  # (B, 3, 3)
        T_hand = self.translations_hand  # (B, 1, 3)
        rots = R_o2h.unsqueeze(0) @ R_hand.unsqueeze(1)
        transl = torch.add(
                torch.matmul(
                    T_o2h.unsqueeze(0),  # (N, 1, 3) -> (1, N, 1, 3)
                    R_hand.unsqueeze(1),  # (B, 3, 3) -> (B, 1, 3, 3)
                ),  # (B, N, 1, 3)
                T_hand.unsqueeze(1),  # (B, 1, 3) -> (B, 1, 1, 3)
            ) # (B, N, 1, 3)
        return torch.matmul(
            self.verts_object_og * scale, rots) + (scale * transl)
    

class HOForwarderV2Impl(HOForwarderV2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def loss_sil_hand(self, compute_iou=False, func='iou'):
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
        rotmat = self.rot_mat_hand
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
    
    def forward_hand(self,
                     loss_weights={
                         'sil': 1,
                         'pca': 1,
                         'rot': 1,
                         'transl': 1,
                     }):
        l_sil = self.loss_sil_hand(compute_iou=False, func='iou').sum()
        l_pca = self.loss_pca_interpolation().sum()
        l_rot = self.loss_hand_rot().sum()
        l_transl = self.loss_hand_transl().sum()
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

    """ Object functions """

    def render_obj(self, verts=None) -> torch.Tensor:
        """
        Renders objects according to current rotation and translation.

        Returns:
            images: ndarray (B, N_init, W, W)
        """
        if verts is None:
            verts = self.get_verts_object()  # (B, N, V, 3)
        b = verts.size(0)
        n = verts.size(1)
        batch_faces = self.faces_object  # (B*N, F, 3)
        batch_K = self.camintr.unsqueeze(1).repeat(1, n, 1, 1)  # (B, 3, 3) -> (B, N, 3, 3)
        images = self.renderer(
            verts.view(b*n, -1, 3),
            batch_faces,
            K=batch_K.view(b*n, -1, 3),
            mode='silhouettes')
        images = images.view(b, n, self.mask_size, self.mask_size)
        return images

    def forward_obj_pose_render(self, loss_only=True, func='l2'):
        """ Reimplement the PoseRenderer.foward() 
        
        Returns:
            loss_dict: dict with
                - mask: (N,)
                - offscreen: (N,)
        """
        b, n, w = self.bsize, self.num_obj_init, self.mask_size
        verts = self.get_verts_object()
        image = self.keep_mask_object[:, None] * self.render_obj(verts)  # (B, N, W, W)
        image_ref = self.ref_mask_object[:, None]  # (B, 1, W, W)
        loss_dict = {}
        if func == 'l2':
            loss_mask = torch.sum((image - image_ref)**2, dim=(2, 3)).mean(dim=0)
        elif func == 'iou':
            loss_mask = iou_loss(image, image_ref).mean(dim=0)

        loss_dict["mask"] = loss_mask
        if not loss_only:
            with torch.no_grad():
                iou = batch_mask_iou(
                    image.view(b*n, w, w).detach(),
                    image_ref.repeat(1, n, 1, 1).view(b*n, w, w).detach())  # (B*N,)
                iou = iou.view(b, n).mean(0)  # (N,)
        loss_dict["offscreen"] = 100000 * self.compute_offscreen_loss(verts)
        if not loss_only:
            return loss_dict, iou, image
        else:
            return loss_dict

    def compute_offscreen_loss(self, verts):
        """
        Computes loss for offscreen penalty. This is used to prevent the degenerate
        solution of moving the object offscreen to minimize the chamfer loss.

        Args:
            verts: (B, N, V, 3)

        Returns:
            loss: (N,)
        """
        # On-screen means coord_xy between [-1, 1] and far > depth > 0
        b, n = verts.size(0), verts.size(1)
        batch_K = self.camintr.unsqueeze(1).repeat(1, n, 1, 1)  # (B, N, 3, 3)
        proj = nr.projection(
            verts.view(b*n, -1, 3),
            batch_K.view(b*n, 3, 3),
            self.renderer.R,
            self.renderer.t,
            self.renderer.dist_coeffs,
            orig_size=1,
        )  # (B*N, ...)
        coord_xy, coord_z = proj[:, :, :2], proj[:, :, 2:]
        zeros = torch.zeros_like(coord_z)
        lower_right = torch.max(coord_xy - 1,
                                zeros).sum(dim=(1, 2))  # Amount greater than 1
        upper_left = torch.max(-1 - coord_xy,
                               zeros).sum(dim=(1, 2))  # Amount less than -1
        behind = torch.max(-coord_z, zeros).sum(dim=(1, 2))
        too_far = torch.max(coord_z - self.renderer.far, zeros).sum(dim=(1, 2))
        loss = lower_right + upper_left + behind + too_far  # (B*N)
        return loss.view(b, n).mean(0)
    
    """ Hand-Object interaction """



from typing import List, Tuple
import numpy as np
from libzhifan.geometry import SimpleMesh, visualize_mesh, projection, CameraManager
import matplotlib.pyplot as plt
import cv2


class HOForwarderV2Vis(HOForwarderV2Impl):
    def __init__(self, 
                 vis_rend_size=256, 
                 ihoi_img_patch=None, 
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.vis_rend_size = vis_rend_size
        self.ihoi_img_patch = ihoi_img_patch
    
    def get_meshes(self, scene_idx, obj_idx, **mesh_kwargs) -> Tuple[SimpleMesh]:
        """
        Args:
            scene_idx: index of scene (timestep)
            obj_idx: -1 indicates no object
        
        Returns:
            mhand: SimpleMesh
            mobj: SimpleMesh, or None if obj_idx < 0
        """
        hand_color = mesh_kwargs.pop('hand_color', 'light_blue')
        obj_color = mesh_kwargs.pop('obj_color', 'yellow')
        with torch.no_grad():
            verts_hand = self.get_verts_hand(**mesh_kwargs)[scene_idx]
            mhand = SimpleMesh(
                verts_hand, self.faces_hand[scene_idx], tex_color=hand_color)
            if obj_idx < 0:
                mobj = None
            else:
                verts_obj = self.get_verts_object(**mesh_kwargs)[scene_idx, obj_idx]
                mobj = SimpleMesh(
                    verts_obj, self.faces_object[0], tex_color=obj_color)
        return mhand, mobj

    def to_scene(self, scene_idx=-1, obj_idx=0, 
                 show_axis=False, viewpoint='nr', **mesh_kwargs):
        """ Returns a trimesh.Scene """
        if scene_idx >= 0:
            mhand, mobj = self.get_meshes(
                scene_idx=scene_idx, obj_idx=obj_idx, **mesh_kwargs)
            return visualize_mesh([mhand, mobj],
                                  show_axis=show_axis,
                                  viewpoint=viewpoint)

        """ Render all """
        hand_color = mesh_kwargs.pop('hand_color', 'light_blue')
        obj_color = mesh_kwargs.pop('obj_color', 'yellow')
        with torch.no_grad():
            verts_hand = self.get_verts_hand(**mesh_kwargs)
            verts_obj = self.get_verts_object(**mesh_kwargs)

        meshes = []
        disp = 0.15  # displacement
        for t in range(self.bsize):
            mhand = SimpleMesh(
                verts_hand[t], self.faces_hand[t], tex_color=hand_color)
            mhand.apply_translation_([t * disp, 0, 0])
            mobj = SimpleMesh(
                verts_obj[t, obj_idx], self.faces_object[0], tex_color=obj_color)
            mobj.apply_translation_([t * disp, 0, 0])
            meshes.append(mhand)
            meshes.append(mobj)
        return visualize_mesh(meshes, show_axis=show_axis)

    def render_summary(self, scene_idx, obj_idx=0) -> np.ndarray:
        a1 = np.uint8(self.ihoi_img_patch[scene_idx])
        mask_hand = self.ref_mask_hand[scene_idx].cpu().numpy().squeeze()
        mask_obj = self.ref_mask_object[scene_idx].cpu().numpy()
        all_mask = np.zeros_like(a1, dtype=np.float32)
        all_mask = np.where(
            mask_hand[...,None], (0, 0, 0.8), all_mask)
        all_mask = np.where(
            mask_obj[...,None], (0.6, 0, 0), all_mask)
        all_mask = np.uint8(255*all_mask)
        a2 = cv2.addWeighted(a1, 0.9, all_mask, 0.5, 1.0)
        a3 = np.uint8(self.render_scene(scene_idx=scene_idx, obj_idx=obj_idx)*255)
        b = np.uint8(255*self.render_triview(scene_idx=scene_idx, obj_idx=obj_idx))
        a = np.hstack([a3, a2, a1])
        return np.vstack([a,
                          b])

    def render_scene(self, scene_idx, obj_idx=0, 
                     with_hand=True, **mesh_kwargs) -> np.ndarray:
        """ returns: (H, W, 3) """
        if not with_hand:
            return self.ihoi_img_patch[scene_idx]
        mhand, mobj = self.get_meshes(
            scene_idx=scene_idx, obj_idx=obj_idx, **mesh_kwargs)
        img = projection.perspective_projection_by_camera(
            [mhand, mobj],
            CameraManager.from_nr(
                self.camintr.detach().cpu().numpy()[scene_idx], self.vis_rend_size),
            method=dict(
                name='pytorch3d',
                coor_sys='nr',
                in_ndc=False
            ),
            image=self.ihoi_img_patch[scene_idx],
        )

        # Add mask
        all_mask = np.zeros_like(img, dtype=np.float32)
        mask_hand = self.ref_mask_hand[scene_idx].cpu().numpy().squeeze()
        mask_obj = self.ref_mask_object[scene_idx].cpu().numpy()
        all_mask = np.where(
            mask_hand[...,None], (0, 0, 0.8), all_mask)
        all_mask = np.where(
            mask_obj[...,None], (0.6, 0, 0), all_mask)
        all_mask = np.uint8(255*all_mask)
        img = cv2.addWeighted(np.uint8(img*255), 0.9, all_mask, 0.5, 1.0)
        return img

    def render_triview(self, scene_idx, obj_idx=0, **mesh_kwargs) -> np.ndarray:
        """
        Returns:
            (H, W, 3)
        """
        rend_size = 256
        mhand, mobj = self.get_meshes(scene_idx=scene_idx, obj_idx=obj_idx, **mesh_kwargs)
        front = projection.project_standardized(
            [mhand, mobj],
            direction='+z',
            image_size=rend_size,
            method=dict(
                name='pytorch3d',
                coor_sys='nr',
                in_ndc=False
            )
        )
        left = projection.project_standardized(
            [mhand, mobj],
            direction='+x',
            image_size=rend_size,
            method=dict(
                name='pytorch3d',
                coor_sys='nr',
                in_ndc=False
            )
        )
        back = projection.project_standardized(
            [mhand, mobj],
            direction='-z',
            image_size=rend_size,
            method=dict(
                name='pytorch3d',
                coor_sys='nr',
                in_ndc=False
            )
        )
        return np.hstack([front, left, back])

    def render_grid(self, obj_idx=0, with_hand=True):
        l = self.bsize
        num_cols = 5
        num_rows = (l + num_cols - 1) // num_cols
        fig, axes = plt.subplots(
            nrows=num_rows, ncols=num_cols,
            sharex=True, sharey=True, figsize=(20, 20))
        for cam_idx, ax in enumerate(axes.flat, start=0):
            img = self.render_scene(
                scene_idx=cam_idx, obj_idx=obj_idx, with_hand=with_hand)
            ax.imshow(img)
            ax.set_axis_off()
            if cam_idx == l-1:
                break
        plt.tight_layout()
        return fig