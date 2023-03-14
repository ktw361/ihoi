from hydra.utils import to_absolute_path
from typing import Tuple, List, NamedTuple
from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F
import neural_renderer as nr
from pytorch3d.ops import knn_points, knn_gather
from pytorch3d.transforms import rotation_6d_to_matrix
from torch_scatter import scatter_min

from config.epic_constants import REND_SIZE
from nnutils.handmocap import get_hand_faces
from nnutils.mesh_utils_extra import compute_vert_normals
from homan.contact_prior import get_contact_regions
from homan.homan_ManoModel import HomanManoModel
from homan.ho_utils import compute_transformation_persp
from homan.interactions import scenesdf

from homan.lossutils import (
    compute_ordinal_depth_loss, compute_contact_loss,
    compute_collision_loss,
    iou_loss, rotation_loss_v1, compute_chamfer_distance,
    compute_nearest_dist, find_nearest_vecs)
import roma

from libzhifan.numeric import check_shape
from libzhifan.geometry import BatchCameraManager
from libyana.metrics.iou import batch_mask_iou


__left_mano_model = HomanManoModel(
    to_absolute_path("externals/mano"), side='left', pca_comps=45)  # Note(zhifan): in HOMAN num_pca = 16
__right_mano_model = HomanManoModel(
    to_absolute_path("externals/mano"), side='right', pca_comps=45)
_mano_model_dict = {
    'left': __left_mano_model, 'right': __right_mano_model}


class LiteHandModule(nn.Module):
    
    # Input to this module
    class LiteHandParams(NamedTuple):
        """ As ho_forwarder_v2.py:
            mano_trans init to zero,
            mano_rot init to zero,
            scale_hand init to one
        """
        camintr: torch.Tensor  # (L, 3, 3) Ihoi bounding box camera.
        rotations_hand: torch.Tensor  # (L, 6)
        translations_hand: torch.Tensor
        hand_side: str
        mano_pca_pose: torch.Tensor  # (L, 45)
        mano_betas: torch.Tensor  # (L, 10)
        target_masks_hand: torch.Tensor  # (L, W, W), W=256

    # Output of this module
    class HandData(NamedTuple):
        """ ALl these represent different T*N hand data."""
        camintr: torch.Tensor   # (N*T, 3, 3)  Ihoi bounding box camera.
        rotations_hand: torch.Tensor  # (N*T, 6)
        translations_hand: torch.Tensor  # (N*T, 1, 3)
        v_hand_global: torch.Tensor  # (N*T, V, 3)
        v_hand_local: torch.Tensor  # (N*T, V, 3)
        rot_mat_hand: torch.Tensor  # (N*T, 3, 3)
        """ for visualization """
        faces_hand: torch.Tensor  # (N*T, F, 3)
        ref_mask_hand: torch.Tensor  # (N*T, W, W)

        def __getitem__(self, inds):
            return LiteHandModule.HandData(
                camintr=self.camintr[inds], rotations_hand=self.rotations_hand[inds],
                translations_hand=self.translations_hand[inds],
                v_hand_global=self.v_hand_global[inds], v_hand_local=self.v_hand_local[inds],
                rot_mat_hand=self.rot_mat_hand[inds], faces_hand=self.faces_hand[inds],
                ref_mask_hand=self.ref_mask_hand[inds])

    def __init__(self):
        super().__init__()
        self.mask_size = REND_SIZE

        self.renderer = nr.renderer.Renderer(
            image_size=self.mask_size,
            K=None,
            R=torch.eye(3, device='cuda')[None],
            t=torch.zeros([1, 3], device='cuda'),
            orig_size=1)
        self.renderer.light_direction = [1, 0.5, 1]
        self.renderer.light_intensity_direction = torch.as_tensor(0.3)
        self.renderer.light_intensity_ambient = 0.5
        self.renderer.background_color = [1.0, 1.0, 1.0]

    @staticmethod
    def gather0d(source, indices: torch.Tensor):
        """ 
        Args:
            source: (L, F1, F2, ...)
            indices: (N*T)
        Returns:
            (N*T, F1, F2, ...)
        """
        feature_shape = source.shape[1:]
        dummy_shape = (1,) * len(feature_shape)
        N, T = indices.shape
        indices = indices.view(-1, *dummy_shape).expand(-1, *feature_shape)
        out = torch.gather(source, dim=0, index=indices)
        return out
    
    def __getitem__(self, indices) -> HandData:
        """ 
        Args:
            indices: (N, T)

        Returns: a HandData object 
            where it holds B*N different hand data.
        """
        # This function should only copy the member data DIRECTLY.
        # transformation of hand data should not be done here. 
        indices = torch.as_tensor(indices, device=self.rotations_hand.device)
        v_hand_global = self.get_verts_hand()
        v_hand_local = self.get_verts_hand(hand_space=True)
        camintr = self.gather0d(self.camintr, indices)
        rotations_hand= self.gather0d(self.rotations_hand, indices)
        translations_hand = self.gather0d(self.translations_hand, indices)
        v_hand_global = self.gather0d(v_hand_global, indices)
        v_hand_local = self.gather0d(v_hand_local, indices)
        rot_mat_hand = self.gather0d(self.rot_mat_hand, indices)
        faces_hand = self.gather0d(self.faces_hand, indices)
        ref_mask_hand = self.gather0d(self.ref_mask_hand, indices)
        return self.HandData(
            camintr=camintr, rotations_hand=rotations_hand,
            translations_hand=translations_hand,
            v_hand_global=v_hand_global, v_hand_local=v_hand_local,
            rot_mat_hand=rot_mat_hand, faces_hand=faces_hand,
            ref_mask_hand=ref_mask_hand)
    
    @property
    def rot_mat_hand(self) -> torch.Tensor:
        """ (L, 3, 3) matrix, apply to col-vector """
        return rotation_6d_to_matrix(self.rotations_hand)

    def get_verts_hand(self, detach_scale=False, hand_space=False) -> torch.Tensor:
        """
        Args:
            hand_space: if True, return hand vertices in hand space itself.
        """
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
        if hand_space:
            return verts_hand_og

        scale = self.scale_hand.detach() if detach_scale else self.scale_hand
        rotations_hand = self.rot_mat_hand

        return compute_transformation_persp(
            meshes=verts_hand_og,
            translations=self.translations_hand,
            rotations=rotations_hand,
            intrinsic_scales=scale,
        )

    def set_hand_params(self, params: LiteHandParams):
        """ Inititalize person parameters 
        """
        num_source = len(params.camintr)
        camintr = params.camintr
        rotations_hand = params.rotations_hand
        translations_hand = params.translations_hand
        hand_side = 'left' if 'left' in params.hand_side else 'right'
        mano_pca_pose = params.mano_pca_pose
        mano_betas = params.mano_betas
        target_masks_hand = params.target_masks_hand
        mano_trans = torch.zeros([num_source, 3], device=mano_pca_pose.device)
        mano_rot = torch.zeros([num_source, 3], device=mano_pca_pose.device)
        
        self.register_buffer("camintr", camintr)
        self.hand_sides = [hand_side]
        self.hand_nb = 1

        self.mano_model = _mano_model_dict[hand_side]
        translation_init = translations_hand.detach().clone()
        self.translations_hand = nn.Parameter(translation_init,
                                              requires_grad=True)
        rotations_hand = rotations_hand.detach().clone()
        if rotations_hand.shape[-1] == 3:
            raise ValueError('Invalid Input')
        self.rotations_hand = nn.Parameter(rotations_hand, requires_grad=True)

        self.mano_pca_pose = nn.Parameter(mano_pca_pose, requires_grad=True)
        self.mano_rot = nn.Parameter(mano_rot, requires_grad=True)
        self.mano_trans = nn.Parameter(mano_trans, requires_grad=True)
        self.mano_betas = nn.Parameter(torch.zeros_like(mano_betas),
                                        requires_grad=True)
        self.scale_hand = nn.Parameter(
            torch.ones(1).float(),
            requires_grad=True)

        faces_hand = get_hand_faces(hand_side)
        num_faces_hand = faces_hand.size(1)
        self.register_buffer(
            "textures_hand",
            torch.ones(num_source, num_faces_hand, 1, 1, 1, 3))
        self.register_buffer(
            "faces_hand", faces_hand.expand(num_source, num_faces_hand, 3))
        self.cuda()

        self.register_buffer("ref_mask_hand", (target_masks_hand > 0).float())
        self.register_buffer("keep_mask_hand",
                             (target_masks_hand >= 0).float())
        mask_h, mask_w = target_masks_hand.shape[-2:]
        self.register_buffer(
            "masks_human",
            target_masks_hand.view(num_source, 1, mask_h, mask_w).bool())
        self.cuda()
        self._check_shape_hand(num_source)

    def _check_shape_hand(self, bsize):
        check_shape(self.faces_hand, (bsize, -1, 3))
        check_shape(self.camintr, (bsize, 3, 3))
        check_shape(self.ref_mask_hand, (bsize, self.mask_size, self.mask_size))
        mask_shape = self.ref_mask_hand.shape
        check_shape(self.keep_mask_hand, mask_shape)
        check_shape(self.rotations_hand, (bsize, 6))
        check_shape(self.translations_hand, (bsize, 1, 3))
        # ordinal loss
        check_shape(self.masks_human, (bsize, 1, self.mask_size, self.mask_size))

    def forward_hand(self,
                     loss_weights={
                         'sil': 1,
                         'pca': 1,
                         'rot': 10,
                         'transl': 1,
                     }) -> Tuple[torch.Tensor, dict]:
        l_sil = self.loss_sil_hand(compute_iou=False, func='l2').sum()
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

    def loss_pca_interpolation(self) -> torch.Tensor:
        """
        Interpolation Prior: pose(t) = (pose(t+1) + pose(t-1)) / 2

        Returns: (L-2,)
        """
        target = (self.mano_pca_pose[2:] + self.mano_pca_pose[:-2]) / 2
        pred = self.mano_pca_pose[1:-1]
        loss = torch.sum((target - pred)**2, dim=(1))
        return loss

    def loss_hand_rot(self) -> torch.Tensor:
        """ Interpolation loss for hand rotation """
        device = self.rotations_hand.device
        rotmat = self.rot_mat_hand
        # with torch.no_grad():
        rot_mid = roma.rotmat_slerp(
            rotmat[2:], rotmat[:-2],
            torch.as_tensor([0.5], device=device))[0]
        loss = rotation_loss_v1(rot_mid, rotmat[1:-1])
        return loss

    def loss_hand_transl(self) -> torch.Tensor:
        """
        Returns: (L-2,)
        """
        interp = (self.translations_hand[2:] + self.translations_hand[:-2]) / 2
        pred = self.translations_hand[1:-1]
        loss = torch.sum((interp - pred)**2, dim=(1, 2))
        return loss

    def loss_sil_hand(self, compute_iou=False, func='l2'):
        """ returns: (B,) """
        rend = self.renderer(
            self.get_verts_hand(),
            self.faces_hand,
            K=self.camintr,
            mode="silhouettes")
        image = self.keep_mask_hand * rend
        if func == 'l2':
            loss_sil = torch.sum(
                (image - self.ref_mask_hand)**2, dim=(1, 2))
            loss_sil = loss_sil / self.keep_mask_hand.sum(dim=(1,2))
        elif func == 'iou':
            loss_sil = iou_loss(image, self.ref_mask_hand)
        elif func == 'l2_iou':
            loss_sil = torch.sum(
                (image - self.ref_mask_hand)**2, dim=(1, 2))
            loss_sil = loss_sil / self.keep_mask_hand.sum(dim=(1,2))
            with torch.no_grad():
                iou_factor = iou_loss(image, self.ref_mask_hand, post='rev')
            loss_sil = loss_sil * iou_factor

        # loss_sil = loss_sil / self.bsize
        if compute_iou:
            ious = batch_mask_iou(image, self.ref_mask_hand)
            return loss_sil, ious
        return loss_sil


class MVHO(nn.Module):
    """ Multiple-View Hand-Object 
    N*T different hand <pose, verts>, N*T target obj data,
    { (N1, T1), (N1, T2), (N1, T3),
      (N2, T3), (N2, T4), (N2, T5),
      (N3, T6), (N3, T7), (N3, T8) }
    }
    only N init obj poses.

    Almost every internal computation is done with B*N, 
    rather than (B, N) taking two dimensions.
    This works naturally with nr_renderer, though the performance improvement is not clear.
    """

    def __init__(self):
        """
        """
        super().__init__()
        self.num_inits = None
        self.train_size = None
        self.mask_size = REND_SIZE
        self.contact_regions = get_contact_regions()

        """ Set-up silhouettes renderer """
        self.renderer = nr.renderer.Renderer(
            image_size=self.mask_size,
            K=None,
            R=torch.eye(3, device='cuda')[None],
            t=torch.zeros([1, 3], device='cuda'),
            orig_size=1)
        self.renderer.light_direction = [1, 0.5, 1]
        self.renderer.light_intensity_direction = torch.as_tensor(0.3)
        self.renderer.light_intensity_ambient = 0.5
        self.renderer.background_color = [1.0, 1.0, 1.0]
    
    def set_size(self, num_inits, train_size):
        """ Set num_init and train_size 

        Args:
            num_inits: number of initial object poses, actually should be Np parallel
            train_size: number of training samples
        """
        self.num_inits = num_inits
        self.train_size = train_size

    def set_hand_data(self, hand_data: LiteHandModule.HandData):
        """ Set hand data (output) """
        self.register_buffer(
            'camintr',
            hand_data.camintr.detach().clone().requires_grad_(False))
        self.rotations_hand = nn.Parameter(
            hand_data.rotations_hand.detach().clone(), requires_grad=True)
        self.translations_hand = nn.Parameter(
            hand_data.translations_hand.detach().clone(), requires_grad=True)
        self.register_buffer(
            'v_hand',
            hand_data.v_hand_global.detach().clone().requires_grad_(False))
        self.faces_hand = hand_data.faces_hand
        self.ref_mask_hand = hand_data.ref_mask_hand
    
    @property
    def rot_mat_hand(self) -> torch.Tensor:
        """ (N*T 3, 3) matrix, apply to col-vector """
        return rotation_6d_to_matrix(self.rotations_hand)

    """ Object functions """

    def register_obj_buffer(self, 
                            verts_object_og,
                            faces_object,
                            scale_mode):
        """
        Args:
            verts_object_og: (V, 3)
            faces_object: (F, 3)
            scale_mode: str, one of {'depth', 'scalar', 'xyz'}
                but use scale_init if provided.
        """
        self.scale_mode = scale_mode
        self.register_buffer("verts_object_og", verts_object_og)
        """ Do not attempt to copy tensor too early, which will get OOM. """
        self.register_buffer(
            "faces_object", faces_object)
        self.register_buffer(
            "textures_object",
            torch.ones(faces_object.shape[0], 1, 1, 1, 3))

    def set_obj_transform(self,
                          translations_object,
                          rotations_object,
                          scale_object):
        """ Initialize / Re-set object parameters
        N obj poses will be lifted to N*T obj poses

        Args:
            translations_object: (N, 1, 3)
            rotations_object: (N, 6)
            scale_object: 
                -(N, 3) for scale_mode == 'xyz'
                -(N,) 
        """
        if rotations_object.shape[-1] == 3:
            raise ValueError("Invalid Input")
        else:
            rotations_object6d = rotations_object
        self.rotations_object = nn.Parameter(
            rotations_object6d.detach().clone(), requires_grad=True)
        self.translations_object = nn.Parameter(
            translations_object.view(-1, 1, 3).detach().clone(),
            requires_grad=True)
        """ Translation is also a function of scale T(s) = s * T_init """
        self.scale_object = nn.Parameter(
            scale_object.detach().clone(),
            requires_grad=True)

    def set_obj_part(self, part_verts: List):
        """ Object Part(s) define the prior contact region on the object surface.

        Args:
            part_verts: list of int, vertices indices
        """
        self.obj_part_verts = part_verts

    def set_obj_target(self, target_masks_object: torch.Tensor,
                       check_shape=True):
        """
        Args:
            target_masks_object: (N*T, W, W)
        """
        target_masks_object = target_masks_object
        self.register_buffer("ref_mask_object",
                             (target_masks_object > 0).float())
        self.register_buffer("keep_mask_object",
                             (target_masks_object >= 0).float())
        self.cuda()
        if check_shape:
            self._check_shape_object()

    def _check_shape_object(self):
        num_inits = self.num_inits
        check_shape(self.verts_object_og, (-1, 3))
        check_shape(self.faces_object, (-1, 3))
        check_shape(self.ref_mask_object,
            (self.num_inits * self.train_size, self.mask_size, self.mask_size))
        mask_shape = self.ref_mask_object.shape
        check_shape(self.keep_mask_object, mask_shape)
        check_shape(self.rotations_object, (num_inits, 6))
        check_shape(self.translations_object, (num_inits, 1, 3))
        if self.scale_mode == 'xyz':
            check_shape(self.scale_object, (num_inits, 3))
        else:
            check_shape(self.scale_object, (num_inits,))

    def _expand_obj_faces(self, n, t) -> torch.Tensor:
        """
        e.g. self._expand_obj_faces(n, t) -> shape (n*t, f, 3)
        Args:

        Returns:
            obj_faces: (n * t, F_o, 3)
        """
        num_obj_faces = self.faces_object.size(0)
        return self.faces_object.expand(n*t, num_obj_faces, 3)

    @property
    def rot_mat_obj(self) -> torch.Tensor:
        """ (N, 3, 3) matrix which can apply to col-vector 
        each one of N apply to T hand poses
        """
        return rotation_6d_to_matrix(self.rotations_object)
    
    def get_obj_transform_world(self, hand_space=False) -> Tuple:
        """ compose N obj pose to N*T hand poses
        Returns:
            rots: (N*T, 3, 3) apply to col-vec
            transl: (N*T, 1, 3)
            scale: (N*T)
        """
        N = self.num_inits
        T = self.train_size
        R_o2h = self.rot_mat_obj  # (N, 3, 3)
        T_o2h = self.translations_object  # (N, 1, 3)
        scale = self.scale_object

        R_o2h = R_o2h.view(N, 1, 3, 3).expand(N, T, 3, 3).reshape(N*T, 3, 3)
        T_o2h = T_o2h.view(N, 1, 3).expand(N, T, 3).reshape(N*T, 1, 3)
        scale = scale.view(N, 1).expand(N, T).reshape(N*T)
        """ Compound T_o2c (T_obj w.r.t camera) = T_h2c x To2h_ """
        R_hand = self.rot_mat_hand  # (N*T, 3, 3)
        T_hand = self.translations_hand  # (N*T, 1, 3)
        R_o2h_row = R_o2h.permute(0, 2, 1)
        R_hand_row = R_hand.permute(0, 2, 1)  # (N*T, 3, 3) row-vec
        if hand_space:
            transl = T_o2h
            if self.scale_mode == 'depth':
                transl = scale * transl
            return R_o2h, transl, scale
        rots_row = R_o2h_row @ R_hand_row  
        rots = rots_row.permute(0, 2, 1)
        transl = torch.add(
                torch.matmul(
                    T_o2h,  # (N*T, 1, 3)
                    R_hand_row,  # (N*T, 3, 3)
                ),  # (N*T, 1, 3)
                T_hand,  # (N*T, 1, 3)
            ) # (N*T, 1, 3)
        if self.scale_mode == 'depth':
            transl = scale * transl
        elif self.scale_mode == 'scalar':
            transl = transl
        elif self.scale_mode == 'xyz':
            transl = transl
        else:
            raise ValueError("scale_mode not understood")
        return rots, transl, scale

    def get_verts_object(self, hand_space=False) -> torch.Tensor:
        """
            V_out = (V_model x R_o2h + T_o2h) x R_hand + T_hand
                  = V x (R_o2h x R_hand) + (T_o2h x R_hand + T_hand)
        where self.rotations/translations_object is R/T_o2h from object to hand

        Args:
            hand_space: bool, if False, return in camera space

        Returns:
            verts_object: (N*T, V, 3)
        """
        rots, transl, scale = self.get_obj_transform_world(
            hand_space=hand_space)

        verts_obj = self.verts_object_og
        nt = self.num_inits * self.train_size
        verts_obj = verts_obj.view(1, -1, 3).expand(
            nt, -1, -1)
        if self.scale_mode == 'xyz':
            scale = scale.view(-1, 1, 3).expand(nt, -1, -1)
        else:
            scale = scale.view(-1, 1, 1).expand(nt, -1, -1)
        rots_row = rots.permute(0, 2, 1)
        return torch.matmul(verts_obj * scale, rots_row) + transl


class MVHOImpl(MVHO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    """ Object functions """

    def render_obj(self, verts) -> torch.Tensor:
        """
        Renders objects according to current rotation and translation.

        Args:
            verts: (N*, V, 3)

        Returns:
            images: ndarray (N*T, W, W)
        """
        batch_faces = self._expand_obj_faces(self.num_inits, self.train_size)  # (N*T, F, 3)
        batch_K = self.camintr

        images = self.renderer(
            verts, batch_faces, K=batch_K, mode='silhouettes')
        return images

    def forward_obj_pose_render(self,
                                v_obj,
                                loss_only=True,
                                func='l2') -> dict:
        """ Reimplement the PoseRenderer.foward()

        Args:
            v_obj (torch.Tensor): (N*T, V, 3)

        Returns:
            loss_dict: dict with
                - mask: (N*T)
                - offscreen: (N*T)
            iou: (N*T)
        """
        n, t, w = self.num_inits, self.train_size, self.mask_size

        image = self.render_obj(v_obj)  # (N*T, W, W)
        keep = self.keep_mask_object  # (N*T, W, W)
        image = keep * image
        image_ref = self.ref_mask_object  # (N*T, W, W)

        loss_dict = {}
        if func == 'l2':
            loss_mask = torch.sum((image - image_ref)**2, dim=(-2,-1))
            loss_mask = loss_mask / (n*image_ref.sum(dim=(-2,-1)))
        elif func == 'iou':
            loss_mask = iou_loss(image, image_ref)
        elif func == 'l2_iou':
            loss_mask = torch.sum(
                (image - image_ref)**2, dim=(-2,-1))
            loss_mask = loss_mask / (n*image_ref.sum(dim=(-2,-1)))
            with torch.no_grad():
                iou_factor = iou_loss(image, image_ref, post='rev')
            loss_mask = loss_mask * iou_factor

        loss_dict["mask"] = loss_mask
        if not loss_only:
            with torch.no_grad():
                iou = batch_mask_iou(
                    image.detach(),
                    image_ref.detach())  # (N*T,)
        loss_dict["offscreen"] = 100000 * self.compute_offscreen_loss(
            v_obj)
        if not loss_only:
            return loss_dict, iou, image
        else:
            return loss_dict

    def compute_offscreen_loss(self, verts):
        """
        Computes loss for offscreen penalty. This is used to prevent the degenerate
        solution of moving the object offscreen to minimize the chamfer loss.

        Args:
            verts: (N*, V, 3)
            camintr: (N*, 3, 3)

        Returns:
            loss: (N*T)
        """
        # On-screen means coord_xy between [-1, 1] and far > depth > 0
        n, t = self.num_inits, self.train_size
        batch_K = self.camintr
        proj = nr.projection(
            verts,
            batch_K,
            self.renderer.R,
            self.renderer.t,
            self.renderer.dist_coeffs,
            orig_size=1,
        )  # (N*T, ...)
        coord_xy, coord_z = proj[:, :, :2], proj[:, :, 2:]
        zeros = torch.zeros_like(coord_z)
        lower_right = torch.max(coord_xy - 1,
                                zeros).sum(dim=(1, 2))  # Amount greater than 1
        upper_left = torch.max(-1 - coord_xy,
                               zeros).sum(dim=(1, 2))  # Amount less than -1
        behind = torch.max(-coord_z, zeros).sum(dim=(1, 2))
        too_far = torch.max(coord_z - self.renderer.far, zeros).sum(dim=(1, 2))
        loss = lower_right + upper_left + behind + too_far  # (N*T)
        return loss

    """ Hand-Object interaction """

    def loss_ordinal_depth(self, v_hand=None, v_obj=None):
        raise NotImplementedError
        v_hand = self.get_verts_hand() if v_hand is None else v_hand
        v_obj = self.get_verts_object() if v_obj is None else v_obj
        b = v_obj.size(0)

        _, depths_o, silhouettes_o = self.renderer.render(
            v_obj[:, 0], self._expand_obj_faces(b),
            self.textures_object.expand(b, *self.textures_object.shape))  # (B, 256, 256)
        silhouettes_o = (silhouettes_o == 1).bool()

        _, depths_p, silhouettes_p = self.renderer.render(
            v_hand, self.faces_hand, self.textures_hand)
        silhouettes_p = (silhouettes_p == 1).bool()

        all_masks = torch.stack(
            [self.ref_mask_object.bool(), self.ref_mask_hand.bool()],
            axis=1)
        silhouettes = [silhouettes_o, silhouettes_p]
        depths = [depths_o, depths_p]
        loss_dict = compute_ordinal_depth_loss(
            all_masks, silhouettes, depths)
        return loss_dict['depth']

    def physical_factor(self) -> torch.Tensor:
        """ We should relate 3D distance to render image size
        so they have similar magnitude.

        d_pixel = d_3d * factor
        loss = d_pixel**2

        Returns:
            factor : (B,) same length as sample_indces
        """
        fx = self.camintr[:, 0, 0]
        return fx * self.mask_size

    def loss_chamfer(self,
                     obj_idx=0,
                     v_hand=None,
                     v_obj=None) -> torch.Tensor:
        """ returns a scalar """
        raise NotImplementedError
        v_hand = self.get_verts_hand() if v_hand is None else v_hand
        v_obj = self.get_verts_object() if v_obj is None else v_obj
        l_chamfer = compute_chamfer_distance(
            v_hand, v_obj[:, obj_idx],
            batch_reduction=None)  # (B,)
        l_chamfer = self.physical_factor() * l_chamfer
        l_chamfer = (l_chamfer**2).sum()
        return l_chamfer

    def loss_nearest_dist(self, v_hand=None, v_obj=None, phy_factor=True) -> torch.Tensor:
        """ 
        Args:
            v_hand: (N*T, V, 3)
            v_obj: (N*, V, 3)
        returns (N*T) 
        """
        v_hand = self.v_hand if v_hand is None else v_hand
        v_obj = self.get_verts_object() if v_obj is None else v_obj
        l_min_d = compute_nearest_dist(v_obj, v_hand)
        if phy_factor:
            phy_factor = self.physical_factor()
            l_min_d = l_min_d * phy_factor
        return l_min_d

    def max_min_dist(self, v_hand=None, v_obj=None) -> float:
        """ max of min_dist over temporal dimension
        Args:
            v_hand: (N*T, V, 3)
            v_obj: (N*, V, 3)
        """
        v_hand = self.v_hand if v_hand is None else v_hand
        v_obj = self.get_verts_object() if v_obj is None else v_obj
        max_min_d = compute_nearest_dist(v_obj, v_hand).max()
        return max_min_d.item()

    def loss_collision(self,
                       v_hand=None, v_obj=None, phy_factor=True):
        """
        Returns:
            (N*T)
        """
        v_hand = self.v_hand if v_hand is None else v_hand
        v_obj = self.get_verts_object() if v_obj is None else v_obj

        n, t = self.num_inits, self.train_size
        l_collision = compute_collision_loss(
            v_hand, v_obj, self._expand_obj_faces(n, t))
        if phy_factor:
            l_collision = self.physical_factor() * l_collision
        return l_collision
    
    def compute_hand_iou(self, v_hand=None, homan=None):
        """ Borrowing keep_mask_hand from post!
        Args:
            homan: HOForwarderV2 which stores the same images as this class
        """
        v_hand = self.v_hand if v_hand is None else v_hand
        rend = self.renderer(
            v_hand,
            self.faces_hand,
            K=self.camintr,
            mode="silhouettes")
        image = homan.keep_mask_hand * rend

        ious = batch_mask_iou(image, homan.ref_mask_hand)
        return ious

    def penetration_depth(self, h2o_only=True) -> float:
        """
        Max penetration depth over all frames,
        report in mm
        Returns:
            - hand into object
            - object into hand
        """
        f_hand = self.faces_hand[0]
        f_obj = self.faces_object
        v_hand = self.v_hand
        v_obj = self.get_verts_object()

        sdfl = scenesdf.SDFSceneLoss([f_hand, f_obj])
        sdf_loss, sdf_meta = sdfl([v_hand, v_obj])
        # max_depths = sdf_meta['dist_values'][(1, 0)].max(1)[0]
        h_to_o = sdf_meta['dist_values'][(1, 0)].max(1)[0].max().item()
        o_to_h = sdf_meta['dist_values'][(0, 1)].max(1)[0].max().item()
        h_to_o = h_to_o * 1000
        o_to_h = o_to_h * 1000
        if h2o_only:
            return h_to_o
        else:
            return h_to_o, o_to_h

    """ Contact regions """

    def loss_obj_upright(self) -> torch.Tensor:
        """ Assume z-axis point inward
        R_world * (0, 0, 1)' -> (0, 0, -1)'
        
        Returns:
            (B,)
        """
        raise NotImplementedError
        rots, _, _ = self.get_obj_transform_world()  # (B, 1, 3, 3)
        rots_target = torch.tensor([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]], device=rots.device, dtype=rots.dtype)
        rots_target = rots_target.view(1, 1, 3, 3).expand_as(rots)
        # What is the best differentiable function to measure two rotations?
        loss = ( (rots - rots_target).pow(2) ).sum([1, 2])
        return loss

    def loss_closeness(self,
                       v_hand=None, v_obj=None, v_obj_select=None,
                       squared_dist=False,
                       num_priors=8, 
                       reduce_type='avg',
                       num_nearest_points=1):
        """
        L = distance from finger tips to their nearest vertices
        average over 8(=5+3) regions.
        Options:
            (5 regions vs 8 regions) x (min vs avg)

        Args:
            v_hand: (N*T, V, 3)
            v_obj: (N*T, V, 3)
            squared_dist: whether to calc loss as squared distance
            num_priors: 5 or 8
            reduce: 'min' or 'avg'

        Returns:
            loss: (N*T)
        """
        k1 = num_nearest_points
        v_hand = self.v_hand if v_hand is None else v_hand
        v_obj = self.get_verts_object() if v_obj is None else v_obj

        n, t, obj_size = self.num_inits, self.train_size, v_obj.size(-2)
        vn_obj = compute_vert_normals(v_obj, faces=self.faces_object)
        if v_obj_select is not None:
            v_obj = v_obj[:, v_obj_select, :]
            vn_obj = vn_obj[:, v_obj_select, :]

        ph_idx = reduce(lambda a, b: a + b, self.contact_regions.verts, [])
        ph = v_hand[:, ph_idx, :]  # (N*T, CONTACT, 3)
        _, idx, nn = knn_points(ph, v_obj, K=k1, return_nn=True)
        # idx: (N*T, CONTACT, k1),  nn: (N*T, CONTACT, k1, 3)
        vn_obj_nn = knn_gather(vn_obj, idx)  # (N*T, CONTACT, k1, 3)

        ph = ph.view(n*t, -1, 1, 3).expand(-1, -1, k1, -1)
        prod = torch.sum((ph - nn) * vn_obj_nn, dim=-1)  # (N*T, CONTACT, k1, 3) => (N*T, CONTACT, k1)
        prod = prod**2 if squared_dist else prod.abs_()
        index = torch.cat(
            [prod.new_zeros(len(v), dtype=torch.long) + i
             for i, v in enumerate(self.contact_regions.verts)])
        
        # Use mean for k1 nearest points
        prod = prod.mean(-1)    # (N*T, CONTACT, k1) => (N*T, CONTACT)
        regions_min, _ = scatter_min(src=prod, index=index, dim=1)  # (N*T, 8)
        regions_min = regions_min[..., :num_priors]

        if reduce_type == 'min':
            loss = regions_min.min(dim=-1).values
        elif reduce_type == 'avg':
            loss = regions_min.mean(dim=-1)

        phy_factor = self.physical_factor()
        # phy_factor = phy_factor.view(-1, 1).expand(-1, num_obj)
        # loss = loss.view(bsize, num_obj)
        loss = loss * phy_factor
        return loss

    def loss_insideness(self,
                        v_hand=None, v_obj=None, v_obj_select=None,
                        squared_dist=False,
                        num_nearest_points=3,
                        debug_viz=False):
        """
        For all p in object, find nearest K points in hand prior regions,
            compute distance (inner product w/ normal) as loss at this p.
            negative indicate Wrong position.

            Loss = \Avg -1.0 * max(loss_p, 0)
        
        Args:
            v_obj_select: List of vertices to compute loss
            num_nearest_points: number of nearest K points in hand

        Returns:
            loss: (N*B)
        """
        k2 = num_nearest_points

        v_hand = self.v_hand if v_hand is None else v_hand
        vn_hand = compute_vert_normals(v_hand, faces=self.faces_hand[0])
        v_obj = self.get_verts_object() if v_obj is None else v_obj
        if v_obj_select is not None:
            v_obj = v_obj[:, v_obj_select, :]

        n, t, v_obj_size = self.num_inits, self.train_size, v_obj.size(-2)
        p_obj = v_obj  # (n*t, V, 3)
        # p_obj = v_obj.view(bsize, num_obj * v_obj_size, 3)

        p2_idx = reduce(lambda a, b: a + b, self.contact_regions.verts, [])
        p2 = v_hand[:, p2_idx, :]
        vn_hand_part = vn_hand[:, p2_idx, :]  # (N*T, CONTACT, 3)

        _, idx, nn = knn_points(p_obj, p2, K=k2, return_nn=True)  
        # idx: (N*T, V, k2), nn: (N*T, V, k2, 3)
        nn_normals = knn_gather(vn_hand_part, idx)  # (N*T, V, k2, 3)

        """ Reshaping """
        p1 = p_obj.view(n*t, v_obj_size, 1, 3).expand(-1, -1, k2, -1)
        nn_normals = nn_normals.view(n*t, v_obj_size, k2, 3)

        vec = (p1 - nn)  # (N*T, V, k2, 3)
        prod = (vec * nn_normals).sum(-1)  # (N*T, V, k2)
        if squared_dist:
            prod = prod**2
        score  = prod.mean(-1)  # (N*T, V)
        loss =  (- score.clamp_max_(0)).mean(-1)  # (N*T)
        phy_factor = self.physical_factor()
        loss = loss * phy_factor

        if debug_viz:
            pose_idx = 0
            scene = 0
            vals = score[scene].detach().cpu().numpy()
            mhand, mobj = self.get_meshes(pose_idx, scene)
            colors = np.where(
                vals[..., None] >= 0,
                mobj.visual.vertex_colors,
                trimesh.visual.interpolate(vals, color_map='jet'))
            mobj.visual.vertex_colors = colors
            return trimesh.Scene([mhand, mobj])

        return loss
    
    """ system objectives """
    def train_loss(self, optim_cfg, print_metric=False):
        """ cfg: `optim` section of the config """
        with torch.no_grad():
            v_hand = self.v_hand  # (N*T, V, 3)
        v_obj = self.get_verts_object()  # (N*T, V, 3)

        l_obj_dict = self.forward_obj_pose_render(
            v_obj=v_obj, func=optim_cfg.obj_sil_func)  # (N*T)
        l_obj_mask = l_obj_dict['mask'].sum()

        if hasattr(optim_cfg, 'obj_part_prior') and optim_cfg.obj_part_prior:
            v_obj_select = self.obj_part_verts
        else:
            v_obj_select = None
        l_inside = self.loss_insideness(
            v_hand=v_hand, v_obj=v_obj, v_obj_select=v_obj_select,
            num_nearest_points=optim_cfg.loss.inside.num_nearest_points)
        l_inside = l_inside.sum()

        l_close = self.loss_closeness(
            v_hand=v_hand, v_obj=v_obj, v_obj_select=v_obj_select,
            num_priors=optim_cfg.loss.close.num_priors,
            reduce_type=optim_cfg.loss.close.reduce,
            num_nearest_points=optim_cfg.loss.close.num_nearest_points)
        l_close = l_close.sum()

        # Accumulate
        tot_loss = optim_cfg.loss.mask.weight * l_obj_mask +\
            optim_cfg.loss.inside.weight * l_inside +\
            optim_cfg.loss.close.weight * l_close

        if print_metric:
            max_min_dist = self.loss_nearest_dist(
                v_hand=v_hand, v_obj=v_obj).max()
            print(
                f"obj_mask:{l_obj_mask.item():.3f} "
                f"inside:{l_inside.item():.3f} "
                f"close:{l_close.item():.3f} "
                f"max_min_dist: {max_min_dist:.3f} "
                )

        return tot_loss
    
    def eval_metrics(self, unsafe=False, avg=False,
                     post_homan=None):
        """ Evaluate metric on ALL frames

        Returns: dict
            -iou: (N*T)
            -max_min_dist: scalar
        """
        with torch.no_grad():
            v_hand = self.v_hand
            v_obj = self.get_verts_object()
            _, oious, _ = self.forward_obj_pose_render(
                v_obj=v_obj, loss_only=False)
            max_min_dist = self.max_min_dist(
                v_hand=v_hand, v_obj=v_obj)
            if unsafe:
                pd_h2o = self.penetration_depth(h2o_only=True)
            if post_homan:
                hious = self.compute_hand_iou(v_hand, homan=post_homan)
                if avg:
                    hious = hious.mean().item()
        if avg:
            oious = oious.mean().item()

        metrics = {
            'oious': oious,
            'max_min_dist': max_min_dist,
        }
        if unsafe:
            metrics['pd_h2o'] = pd_h2o
        if post_homan:
            metrics['hious'] = hious
        return metrics


from typing import List, Tuple
import trimesh
import numpy as np
from libzhifan.geometry import SimpleMesh, visualize_mesh, projection, CameraManager
from libzhifan.geometry import visualize as geo_vis
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('svg')  # seems svg renders plt faster
import cv2


class MVHOVis(MVHOImpl):
    def __init__(self,
                 vis_rend_size=256,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.vis_rend_size = vis_rend_size
    
    def set_ihoi_img_patch(self, ihoi_img_patch):
        self.ihoi_img_patch = ihoi_img_patch

    def get_meshes(self, pose_idx, scene_idx, **mesh_kwargs) -> Tuple[SimpleMesh]:
        """
        Args:
            pose_idx: 
            scene_idx: index of scene (timestep)
                'pose_idx * T + scene_idx' index into N*T

        Returns:
            mhand: SimpleMesh
            mobj: SimpleMesh, or None if obj_idx < 0
        """
        hand_color = mesh_kwargs.pop('hand_color', 'light_blue')
        obj_color = mesh_kwargs.pop('obj_color', 'yellow')
        nt_ind = pose_idx * self.train_size + scene_idx
        with torch.no_grad():
            verts_hand = self.v_hand[nt_ind]
            mhand = SimpleMesh(
                verts_hand, self.faces_hand[nt_ind], tex_color=hand_color)
            # if pose_idx < 0:
            #     mobj = None
            # else:
            verts_obj = self.get_verts_object(**mesh_kwargs)[nt_ind]
            mobj = SimpleMesh(
                verts_obj, self.faces_object, tex_color=obj_color)
        return mhand, mobj

    def finger_with_normals(self, nt_ind,
                            regions=(0,1,2,3,4,5,6,7)) -> trimesh.Scene:
        """
        Returns: a Scene with single hand, finger regions marked with normals.
        """
        hand_color = 'light_blue'
        with torch.no_grad():
            verts_hand = self.v_hand[nt_ind]
            vn = compute_vert_normals(verts_hand, self.faces_hand[nt_ind])
            mhand = SimpleMesh(
                verts_hand, self.faces_hand[nt_ind], tex_color=hand_color)

        paths = []
        for i, v_inds in enumerate(self.contact_regions.verts):
            if i not in regions:
                continue
            geo_vis.color_verts(mhand, v_inds, (255, 0, 0))

            v_parts = verts_hand[v_inds].detach().cpu().numpy()
            vn_parts = vn[v_inds].cpu().numpy()
            vec = np.column_stack(
                (v_parts, v_parts + (vn_parts * mhand.scale * .05)))
            path = trimesh.load_path(vec.reshape(-1, 2, 3))
            paths.append(path)

        return trimesh.Scene([mhand] + paths)

    def visualize_nearest_normals(self,
                                  nt_ind,
                                  display=('hand', 'obj', 'normals'),
                                  regions=5,
                                  ) -> trimesh.Scene:
        """
        Visualize each region's nearest point to the object
        Args:
            nt_ind: index into N*T
        """
        # Find nearest point indices
        k1, k2 = 1, 1
        h_paths = []
        o_paths = []
        vh_inds = []
        vo_inds = []
        with torch.no_grad():
            v_hand = self.v_hand[[nt_ind]]
            v_obj = self.get_verts_object()[[nt_ind]]
            vn_hand = compute_vert_normals(v_hand, faces=self.faces_hand[0])
            vn_obj = compute_vert_normals(v_obj, faces=self.faces_object)
            mhand = SimpleMesh(v_hand[0], self.faces_hand[nt_ind])
            mobj = SimpleMesh(v_obj[0], self.faces_object)
            p2 = v_obj

            for part in self.contact_regions.verts[:regions]:
                p1 = v_hand[:, part, :]
                pn1 = vn_hand[:, part, :]
                v1, v2, vh_ind, vo_ind, vn1, vn2 = find_nearest_vecs(
                    p1, p2, k1=k1, k2=k2, pn1=pn1, pn2=vn_obj)
                """
                To get index in the hand,
                first get index
                """
                vh_ind = vh_ind[nt_ind].squeeze().item()
                vh_ind = part[vh_ind]
                vh_inds.append( vh_ind )
                vo_inds.append( vo_ind[nt_ind].squeeze().item() )

                v1 = v1[nt_ind].cpu().numpy()
                vn1 = vn1[nt_ind].cpu().numpy()
                v2 = v2.squeeze_(1)[nt_ind].cpu().numpy()  # (1, 3)
                vn2 = vn2.squeeze_(1)[nt_ind].cpu().numpy()  # (1, 3)
                vec1 = np.column_stack(
                    (v1, v1 + (vn1 * mhand.scale * 0.05)))
                vec2 = np.column_stack(
                    (v2, v2 + (vn2 * mobj.scale * 0.05)))
                vec1 = trimesh.load_path(vec1.reshape(-1, 2, 3))
                vec2 = trimesh.load_path(vec2.reshape(-1, 2, 3))
                h_paths.append(vec1)
                o_paths.append(vec2)

            geo_vis.color_verts(mhand, vh_inds, (255, 0, 0))
            geo_vis.color_verts(mobj, vo_inds, (0, 0, 255))

        scene_geoms = []
        if 'hand' in display:
            scene_geoms.append(mhand)
        if 'obj' in display:
            scene_geoms.append(mobj)
        if 'normals' in display:
            if 'hand' in display:
                scene_geoms += h_paths
            if 'obj' in display:
                scene_geoms += o_paths
        return trimesh.Scene(scene_geoms)

    def to_scene(self, pose_idx, scene_idx=-1,
                 show_axis=False, viewpoint='nr', 
                 scene_indices=None, disp=0.15, 
                 **mesh_kwargs):
        """ Returns a trimesh.Scene """
        if scene_idx >= 0:
            mhand, mobj = self.get_meshes(
                pose_idx=pose_idx, scene_idx=scene_idx, **mesh_kwargs)
            return visualize_mesh([mhand, mobj],
                                  show_axis=show_axis,
                                  viewpoint=viewpoint)

        """ Render all """
        hand_color = mesh_kwargs.pop('hand_color', 'light_blue')
        obj_color = mesh_kwargs.pop('obj_color', 'yellow')
        with torch.no_grad():
            verts_hand = self.v_hand.view(self.num_inits, self.train_size, -1, 3)
            verts_obj = self.get_verts_object(**mesh_kwargs).view(
                self.num_inits, self.train_size, -1, 3)
            verts_hand = verts_hand[pose_idx]
            verts_obj = verts_obj[pose_idx]

        meshes = []
        scene_indices = scene_indices or range(self.bsize)
        for i, t in enumerate(scene_indices):
            mhand = SimpleMesh(
                verts_hand[t], self.faces_hand[t], tex_color=hand_color)
            mhand.apply_translation_([i * disp, 0, 0])
            mobj = SimpleMesh(
                verts_obj[t], self.faces_object, tex_color=obj_color)
            mobj.apply_translation_([i * disp, 0, 0])
            meshes.append(mhand)
            meshes.append(mobj)
        return visualize_mesh(meshes, show_axis=show_axis, viewpoint=viewpoint)

    def render_summary(self, pose_idx, scene_idx) -> np.ndarray:
        nt_ind = pose_idx * self.train_size + scene_idx
        a1 = np.uint8(self.ihoi_img_patch[nt_ind])
        mask_hand = self.ref_mask_hand[nt_ind].cpu().numpy().squeeze()
        mask_obj = self.ref_mask_object[nt_ind].cpu().numpy()
        all_mask = np.zeros_like(a1, dtype=np.float32)
        all_mask = np.where(
            mask_hand[...,None], (0, 0, 0.8), all_mask)
        all_mask = np.where(
            mask_obj[...,None], (0.6, 0, 0), all_mask)
        all_mask = np.uint8(255*all_mask)
        a2 = cv2.addWeighted(a1, 0.9, all_mask, 0.5, 1.0)
        a3 = np.uint8(self.render_scene(pose_idx=pose_idx, scene_idx=scene_idx)*255)
        b = np.uint8(255*self.render_triview(pose_idx=pose_idx, scene_idx=scene_idx))
        a = np.hstack([a3, a2, a1])
        return np.vstack([a,
                          b])

    def render_scene(self, pose_idx, scene_idx,
                     with_hand=True, overlay_gt=False,
                     **mesh_kwargs) -> np.ndarray:
        """ returns: (H, W, 3) """
        # nt_ind = pose_idx * self.train_size + scene_idx
        nt_ind = scene_idx
        if not with_hand:
            img = self.ihoi_img_patch[nt_ind] / 255
        else:
            mhand, mobj = self.get_meshes(pose_idx=pose_idx,
                scene_idx=scene_idx, **mesh_kwargs)
            if pose_idx < 0:
                mesh_list = [mhand]
            else:
                mesh_list = [mhand, mobj]
            img = projection.perspective_projection_by_camera(
                mesh_list,
                CameraManager.from_nr(
                    self.camintr.detach().cpu().numpy()[nt_ind], self.vis_rend_size),
                method=dict(
                    name='pytorch3d',
                    coor_sys='nr',
                    in_ndc=False
                ),
                image=self.ihoi_img_patch[nt_ind],
            )

        if overlay_gt:
            all_mask = np.zeros_like(img, dtype=np.float32)
            mask_hand = self.ref_mask_hand[nt_ind].cpu().numpy().squeeze()
            all_mask = np.where(
                mask_hand[...,None], (0, 0, 0.8), all_mask)
            # if obj_idx >= 0:
            mask_obj = self.ref_mask_object[nt_ind].cpu().numpy()
            all_mask = np.where(
                mask_obj[...,None], (0.6, 0, 0), all_mask)
            all_mask = np.uint8(255*all_mask)
            img = cv2.addWeighted(np.uint8(img*255), 0.9, all_mask, 0.5, 1.0)
        return img

    def render_triview(self, pose_idx, scene_idx, **mesh_kwargs) -> np.ndarray:
        """
        Returns:
            (H, W, 3)
        """
        rend_size = 256
        mhand, mobj = self.get_meshes(
            pose_idx=pose_idx, scene_idx=scene_idx, **mesh_kwargs)
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

    def render_grid_np(self, pose_idx=0, with_hand=True,
                       *args, **kwargs) -> np.ndarray:
        """ low resolution but faster """
        l = self.train_size
        num_cols = min(l, 5)
        num_rows = (l + num_cols - 1) // num_cols
        imgs = []
        for cam_idx in range(l):
            img = self.render_scene(
                pose_idx=pose_idx,
                scene_idx=cam_idx,
                with_hand=with_hand, *args, **kwargs)
            imgs.append(img)
            if cam_idx == l-1:
                break

        h, w, _ = imgs[0].shape
        out = np.empty(shape=(num_rows*h, num_cols*w, 3), dtype=imgs[0].dtype)
        for row in range(num_rows):
            for col in range(num_cols):
                idx = row*num_cols+col
                if  idx >= l:
                    break
                out[row*h:(row+1)*h, col*w:(col+1)*w, :] = imgs[idx]

        return out

    def render_grid(self, pose_idx=0, with_hand=True,
                    figsize=(10, 10), low_reso=True,
                    return_list=False, *args, **kwargs):
        """
        Args:
            obj_idx: -1 for no object
            with_hand: whether to render hand
            figsize: (w, h)
            low_reso: call render_grid_np() if True
            return_list: return a list of images if True
        """
        if low_reso:
            out = self.render_grid_np(pose_idx=pose_idx, with_hand=with_hand, *args, **kwargs)
            fig, ax = plt.subplots()
            ax.imshow(out)
            plt.axis('off')
            return fig
        
        if return_list:
            ret = []
            for cam_idx in range(self.bsize):
                img = self.render_scene(pose_idx=pose_idx,
                    scene_idx=cam_idx, with_hand=with_hand, *args, **kwargs)
                ret.append(img)
            return ret

        l = self.train_size
        num_cols = 5
        num_rows = (l + num_cols - 1) // num_cols
        fig, axes = plt.subplots(
            nrows=num_rows, ncols=num_cols,
            sharex=True, sharey=True, figsize=figsize)
        for cam_idx, ax in enumerate(axes.flat, start=0):
            if cam_idx > l-1:
                ax.set_axis_off()
                continue
            img = self.render_scene(pose_idx=pose_idx,
                scene_idx=cam_idx, with_hand=with_hand, *args, **kwargs)
            ax.imshow(img)
            ax.set_axis_off()
        plt.tight_layout()
        return fig

    def render_global(self,
                      global_cam: BatchCameraManager,
                      global_images: np.ndarray,
                      pose_idx: int,
                      scene_idx: int,
                      with_hand=True,
                      overlay_gt=False,
                      ) -> np.ndarray:
        """ returns: (H, W, 3) """
        nt_ind = pose_idx * self.train_size + scene_idx
        if not with_hand:
            return self.ihoi_img_patch[nt_ind]
        mhand, mobj = self.get_meshes(pose_idx=pose_idx,
            scene_idx=scene_idx)
        img = projection.perspective_projection_by_camera(
            [mhand, mobj],
            global_cam[scene_idx],
            method=dict(
                name='pytorch3d',
                coor_sys='nr',
                in_ndc=False
            ),
            image=global_images[scene_idx],
        )

        if overlay_gt:
            raise NotImplementedError
            all_mask = np.zeros_like(img, dtype=np.float32)
            mask_hand = self.ref_mask_hand[nt_ind].cpu().numpy().squeeze()
            print(mask_hand.shape, all_mask.shape)
            all_mask = np.where(
                mask_hand[...,None], (0, 0, 0.8), all_mask)
            print(all_mask.shape)
            if obj_idx >= 0:
                mask_obj = self.ref_mask_object[nt_ind].cpu().numpy()
                all_mask = np.where(
                    mask_obj[...,None], (0.6, 0, 0), all_mask)
            all_mask = np.uint8(255*all_mask)
            img = cv2.addWeighted(np.uint8(img*255), 0.9, all_mask, 0.5, 1.0)
        return img

    def make_compare_video(self,
                           global_cam: BatchCameraManager,
                           global_images: np.ndarray,
                           pose_idx: int = 0) -> List[np.ndarray]:
        """
        Args:
            pose_idx: usually 0 for drawing eval frames
        """
        frames = []
        scene_indices = range(self.train_size)

        for i in scene_indices:
            img_mesh = self.render_global(
                global_cam=global_cam,
                global_images=global_images,
                pose_idx=pose_idx,
                scene_idx=i,
                with_hand=True,
                overlay_gt=False)
            img = np.vstack([global_images[i], img_mesh * 255])
            frames.append(img)
        return frames
