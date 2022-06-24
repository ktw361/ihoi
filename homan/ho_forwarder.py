from typing import List
import neural_renderer as nr
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from homan import lossutils
from homan.losses import Losses
from homan.homan_ManoModel import HomanManoModel
from homan.utils.geometry import matrix_to_rot6d, rot6d_to_matrix
from homan.ho_utils import (
    compute_transformation_ortho, compute_transformation_persp)

from libzhifan.geometry import (
    SimpleMesh, visualize_mesh, projection, CameraManager)

from libyana.conversions import npt


""" A re-implemented version of homan_core.py

Translation is inheritly related to scale.
if Y = RX + T, then sY = sRX + sT,
in other word
"""

class HOForwarder(nn.Module):
    def __init__(self,
                 translations_object,  # (1, 1, 3)
                 rotations_object,
                 verts_object_og,
                 faces_object,

                 translations_hand,  # (1, 1, 3)
                 rotations_hand,
                 verts_hand_og,
                 hand_sides,
                 mano_trans,
                 mano_rot,
                 mano_betas,
                 mano_pca_pose,
                 faces_hand,

                 camintr_rois_object,
                 camintr_rois_hand,
                 camintr,
                 target_masks_object,
                 target_masks_hand,
                 class_name='default',
                 cams_hand=None,
                 scale_hand=1.0,
                 scale_object=1.0,
                 optimize_object_scale=False,
                 optimize_ortho_cam=True,
                 hand_proj_mode="persp",
                 optimize_mano=True,
                 optimize_mano_beta=True,
                 inter_type="centroid",
                 image_size:int = 640):
        """
        Hands are received in batch of [h_1_t_1, h_2_t_1, ..., h_1_t_2]
        (h_{hand_index}_t_{time_step})

        Args:
            verts_hand_og: Transformed hands TODO(check)
            target_masks_object: used as ref_mask_obj
            target_masks_hand: used as _ref_mask_hand
            camintr_rois_object: from model.K.detach(),
                and to be used in self.renderer
        """
        super().__init__()

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
        self.optimize_object_scale = optimize_object_scale
        if self.optimize_object_scale:
            # Note we modify translation to be a function of scale:
            # T(s) = s * T_init
            self.scale_object = nn.Parameter(
                scale_object * torch.ones(1).float(),
                requires_grad=True,
            )
        else:
            self.register_buffer("scale_object",
                                 scale_object * torch.ones(1).float())
        self.register_buffer("int_scale_object_mean", torch.ones(1).float())


        """ Inititalize person parameters """
        self.mano_model = HomanManoModel("externals/mano", pca_comps=45)  # TODO(zhifan): was 16
        self.hand_proj_mode = hand_proj_mode
        translation_init = translations_hand.detach().clone()
        self.translations_hand = nn.Parameter(translation_init,
                                              requires_grad=True)
        rotations_hand = rotations_hand.detach().clone()
        self.obj_rot_mult = 1  # This scaling has no effect !
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
                                 torch.ones(1).float() * scale_hand)
        else:
            self.register_buffer("mano_betas", torch.zeros_like(mano_betas))
            self.scale_hand = nn.Parameter(
                scale_hand * torch.ones(1).float(),
                requires_grad=True,
            )
        self.register_buffer("verts_hand_og", verts_hand_og)
        self.register_buffer("int_scale_hand_mean",
                             torch.Tensor([1.0]).float().cuda())
        self.register_buffer("ref_mask_object",
                             (target_masks_object > 0).float())
        self.register_buffer("keep_mask_object",
                             (target_masks_object >= 0).float())
        self.register_buffer("ref_mask_hand", (target_masks_hand > 0).float())
        self.register_buffer("keep_mask_hand",
                             (target_masks_hand >= 0).float())
        self.register_buffer("camintr_rois_object", camintr_rois_object)
        self.register_buffer("camintr_rois_hand", camintr_rois_hand)

        self.register_buffer("faces_object", faces_object)
        self.register_buffer(
            "textures_object",
            torch.ones(faces_object.shape[0], faces_object.shape[1], 1, 1, 1,
                       3))
        self.register_buffer(
            "textures_hand",
            torch.ones(faces_hand.shape[0], faces_hand.shape[1], 1, 1, 1, 3))
        self.register_buffer("faces_hand", faces_hand)
        self.cuda()

        # Setup renderer
        if camintr is None:
            raise ValueError("Not allowed")
        else:
            camintr = npt.tensorify(camintr)
            if camintr.dim() == 2:
                camintr = camintr.unsqueeze(0)
            camintr = camintr.cuda().float()
        rot = torch.cuda.FloatTensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]])
        trans = torch.zeros(1, 3).cuda()
        self.register_buffer("camintr", camintr)
        self.image_size = image_size
        self.renderer = nr.renderer.Renderer(image_size=self.image_size,
                                             K=camintr.clone(),
                                             R=rot,
                                             t=trans,
                                             orig_size=1)
        self.renderer.light_direction = [1, 0.5, 1]
        self.renderer.light_intensity_direction = 0.3
        self.renderer.light_intensity_ambient = 0.5
        self.renderer.background_color = [1.0, 1.0, 1.0]

        mask_h, mask_w = target_masks_hand.shape[-2:]
        self.register_buffer(
            "masks_human", target_masks_hand.reshape(1, 1, mask_h, mask_w).bool())
        self.register_buffer(
            "masks_object", target_masks_object.reshape(1, 1, mask_h, mask_w).bool())

        self.losses = Losses(
            renderer=self.renderer,
            ref_mask_object=self.ref_mask_object,
            keep_mask_object=self.keep_mask_object,
            ref_mask_hand=self.ref_mask_hand,
            keep_mask_hand=self.keep_mask_hand,
            camintr_rois_object=self.camintr_rois_object,
            camintr_rois_hand=self.camintr_rois_hand,
            camintr=self.camintr,
            class_name=class_name,
            hand_nb=self.hand_nb,
            inter_type=inter_type,
        )

    def get_verts_object(self, 
                         translations_object=None,
                         scale_object=None,
                         **kwargs):
        rotations_object = rot6d_to_matrix(self.obj_rot_mult *
                                           self.rotations_object)
        translations_object = self.translations_object \
            if translations_object is None else translations_object
        intrinsic_scales = self.scale_object if scale_object is None else scale_object
        obj_verts = compute_transformation_persp(
            meshes=self.verts_object_og,
            translations=translations_object,
            rotations=rotations_object,
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

    def compute_ordinal_depth_loss(self, return_aux=False, **mesh_kwargs):
        verts_object = self.get_verts_object(**mesh_kwargs)
        verts_hand = self.get_verts_hand(**mesh_kwargs)

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
        if loss_weights is None or loss_weights["lw_pca"] > 0:
            loss_pca = lossutils.compute_pca_loss(self.mano_pca_pose)
            loss_dict.update(loss_pca)
        if loss_weights is None or loss_weights["lw_collision"] > 0:
            # Pushes hand out of object, gradient not flowing through object !
            loss_coll = lossutils.compute_collision_loss(
                verts_hand=verts_hand_det_scale,
                verts_object=verts_object.detach(),
                faces_object=self.faces_object,
                faces_hand=self.faces_hand)
            loss_dict.update(loss_coll)

        if loss_weights is None or loss_weights["lw_contact"] > 0:
            loss_contact, _ = lossutils.compute_contact_loss(
                verts_hand_b=verts_hand_det_scale,
                verts_object_b=verts_object,
                faces_object=self.faces_object,
                faces_hand=self.faces_hand)
            loss_dict.update(loss_contact)
        if loss_weights is None or loss_weights["lw_sil_obj"] > 0:
            sil_loss_dict, sil_metric_dict = self.losses.compute_sil_loss_object(
                verts=verts_object, faces=self.faces_object)
            loss_dict.update(sil_loss_dict)
            metric_dict.update(sil_metric_dict)

        if loss_weights is None or loss_weights['lw_sil_hand'] > 0:
            loss_dict.update(
                self.losses.compute_sil_loss_hand(verts=verts_hand,
                                                  faces=[self.faces_hand] *
                                                  len(verts_hand)))
        if loss_weights is None or loss_weights["lw_inter"] > 0:
            # Interaction acts only on hand !
            if not self.optimize_object_scale:
                inter_verts_object = verts_object.unsqueeze(1).detach()
            else:
                inter_verts_object = verts_object.unsqueeze(1)
            inter_loss_dict, inter_metric_dict = \
                self.losses.compute_interaction_loss(
                verts_hand_b=verts_hand.detach().clone().view(
                    -1, self.hand_nb, 778, 3),
                verts_object_b=inter_verts_object)
            loss_dict.update(inter_loss_dict)
            metric_dict.update(inter_metric_dict)

        if loss_weights is None or loss_weights["lw_scale_obj"] > 0:
            loss_dict[
                "loss_scale_obj"] = lossutils.compute_intrinsic_scale_prior(
                    intrinsic_scales=self.scale_object,
                    intrinsic_mean=self.int_scale_object_mean,
                )
        if loss_weights is None or loss_weights["lw_scale_hand"] > 0:
            loss_dict[
                "loss_scale_hand"] = lossutils.compute_intrinsic_scale_prior(
                    intrinsic_scales=self.scale_hand,
                    intrinsic_mean=self.int_scale_hand_mean,
                )
        if loss_weights is None or loss_weights["lw_depth"] > 0:
            loss_dict.update(self.compute_ordinal_depth_loss(**mesh_kwargs))
        return loss_dict, metric_dict

    def get_meshes(self, **mesh_kwargs) -> List[SimpleMesh]:
        with torch.no_grad():
            verts_obj = self.get_verts_object(**mesh_kwargs)
            verts_hand = self.get_verts_hand(**mesh_kwargs)
            mhand = SimpleMesh(verts_hand, self.faces_hand)
            mobj = SimpleMesh(verts_obj, self.faces_object, tex_color='yellow')
        return mhand, mobj
        
    def to_scene(self, show_axis=False, viewpoint='nr', **mesh_kwargs):
        """ Returns a trimesh.Scene """
        mhand, mobj = self.get_meshes(**mesh_kwargs)
        return visualize_mesh([mhand, mobj],
                              show_axis=show_axis,
                              viewpoint=viewpoint)
    
    def render_scene(self, **mesh_kwargs) -> np.ndarray:
        """ returns: (H, W, 3) """
        mhand, mobj = self.get_meshes(**mesh_kwargs)
        img = projection.perspective_projection_by_camera(
            [mhand, mobj],
            CameraManager.from_nr(
                self.camintr.detach().cpu().numpy(), self.image_size),
            method=dict(
                name='pytorch3d',
                coor_sys='nr',
                in_ndc=False
            ),
            image=None,
            # image=np.uint8(self._image_patch*256),
        )
        return img
    
    @staticmethod
    def pack_homan_kwargs(context, sel_idx=0):
        """
        Returns:
            a dict containing kwargs to initialize this class (HoForwarder)
        """
        pose_machine = context.pose_machine
        obj_bbox = context.obj_bbox
        mask_hand = context.mask_hand
        mask_obj = context.mask_obj

        _ious = pose_machine.pose_model()[1]
        pose_idx = torch.sort(_ious, descending=True).indices[sel_idx].item()
        print(f"pose_idx = {pose_idx}")

        _, target_masks_object, target_masks_hand = pose_machine._get_bbox_and_crop(
            mask_obj, mask_hand, obj_bbox)  # from global to local
        mano_pca_pose = pose_machine.recover_pca_pose()
        mano_rot = torch.zeros([1, 3], device=mano_pca_pose.device)
        mano_trans = torch.zeros([1, 3], device=mano_pca_pose.device)
        camintr_rois_object = pose_machine.pose_model.K  # could be pose_machine.ihoi_cam

        homan_kwargs = dict(
            translations_object = pose_machine.pose_model.translations[[pose_idx]],
            rotations_object = pose_machine.pose_model.rotations_matrix[[pose_idx]],
            verts_object_og = pose_machine.pose_model.vertices,
            faces_object = pose_machine.pose_model.faces[[pose_idx]],
            translations_hand = pose_machine.hand_translation,
            rotations_hand = pose_machine.hand_rotation,
            verts_hand_og = pose_machine.hand_verts,
            hand_sides = ['right'],
            mano_trans = mano_trans,
            mano_rot = mano_rot,
            mano_betas = pose_machine.pred_hand_betas,
            mano_pca_pose = pose_machine.recover_pca_pose(),
            faces_hand = pose_machine.hand_faces,
            
            scale_object = 1.0,
            scale_hand = 1.0,

            camintr_rois_object = camintr_rois_object,
            camintr_rois_hand = camintr_rois_object.clone(),
            camintr = camintr_rois_object.clone(),
            target_masks_hand = torch.as_tensor(target_masks_hand),
            target_masks_object = torch.as_tensor(target_masks_object),

            image_size = pose_machine.rend_size,

            optimize_object_scale = True)

        for k, v in homan_kwargs.items():
            if hasattr(v, 'device'):
                homan_kwargs[k] = v.to('cuda')

        return homan_kwargs
