from typing import NamedTuple
import numpy as np
import torch
import torch.nn as nn
import neural_renderer as nr
from scipy.ndimage.morphology import distance_transform_edt
from functools import cached_property

from obj_pose.utils import compute_pairwise_dist
from obj_pose.cluster_distance_matrix import cluster_distance_matrix
from homan.utils.geometry import rot6d_to_matrix

from libyana.metrics import iou as ioumetrics
from libzhifan.numeric import check_shape


REND_SIZE = 256


class FitResult(NamedTuple):
    verts: torch.Tensor  # (B, N)
    rotations: torch.Tensor  # (N, 3, 3)
    translations: torch.Tensor  # (N, 3)
    loss_dict: dict
    iou: torch.Tensor  # (N)


class PoseRenderer(nn.Module):
    """
    Computes the optimal object pose from an instance mask and an exemplar mesh.
    Closely following PHOSA, we optimize an occlusion-aware silhouette loss
    that consists of a one-way chamfer loss and a silhouette matching loss.
    """

    def __init__(self,
                 ref_image,
                 vertices,
                 faces,
                 rotation_init,
                 translation_init,
                 num_initializations=1,
                 base_rotation=None,
                 base_translation=None,
                 kernel_size=7,
                 camera_K=None,
                 power=0.25,
                 lw_chamfer=0,
                 device='cuda'):
        """
        For B `images`, `base transformations`, `camera_K`
        find N_init `rotations` and `translations`
        for single object.

        Args:
            ref_image: (B, W, W) torch.Tensor float32
            vertices: (V, 3)
            faces: (F, 3)
            rotation_init: (N_init, 3, 3)
            translation_init: (1, 3)
            base_rotation: (B, 3, 3)
            base_translation:  (B, 1, 3)
            camera_K: (B, 3, 3)
                local camera of the object

        """
        assert ref_image.shape[-1] == ref_image.shape[-2], "Must be square."
        super().__init__()
        dtype = vertices.dtype
        device = vertices.device
        self.image_size = ref_image.shape[-1]
        self.bsize = len(camera_K)

        vertices = torch.as_tensor(vertices, device=device)
        faces = torch.as_tensor(faces, device=device)
        self.register_buffer("vertices", vertices)
        self.register_buffer("faces", faces)
        if base_rotation is None:
            base_rotation = torch.eye(
                3, dtype=dtype, device=device).unsqueeze_(0)
        else:
            base_rotation = base_rotation.clone()
        if base_translation is None:
            base_translation = torch.zeros(
                [self.bsize, 1, 3], dtype=dtype, device=device)
        else:
            base_translation = base_translation.clone()
        self.register_buffer('base_rotation', base_rotation)
        self.register_buffer('base_translation', base_translation)

        # Load reference mask.
        # Convention for silhouette-aware loss: -1=occlusion, 0=bg, 1=fg.
        image_ref = (ref_image > 0).float()  # (B, H, H)
        keep_mask = (ref_image >= 0).float()
        self.register_buffer("image_ref", image_ref)
        self.register_buffer("keep_mask", keep_mask)
        self.pool = torch.nn.MaxPool2d(kernel_size=kernel_size,
                                       stride=1,
                                       padding=(kernel_size // 2))
        self.rotations = nn.Parameter(rotation_init.clone().float(),
                                      requires_grad=True)
        if rotation_init.shape[0] != translation_init.shape[0]:
            translation_init = translation_init.repeat(
                num_initializations, 1, 1)
        self.translations = nn.Parameter(translation_init.clone().float(),
                                         requires_grad=True)
        mask_edge = self.compute_edges(image_ref).cpu().numpy()
        edt = distance_transform_edt(1 - (mask_edge > 0))**(power * 2)
        self.register_buffer("edt_ref_edge", torch.from_numpy(edt))
        # Setup renderer.
        if camera_K is None:
            camera_K = torch.FloatTensor([
                [[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]]]).to(device)
        rot = torch.eye(3).unsqueeze(0).to(device)
        trans = torch.zeros(1, 3).to(device)
        self.renderer = nr.renderer.Renderer(
            image_size=ref_image.shape[-1],
            R=rot,  # eye(3)
            t=trans,  # zero
            orig_size=1,
            K=camera_K,
            anti_aliasing=False,
        )
        self.lw_chamfer = lw_chamfer

        self.to(device)
        self._check_shape(self.bsize, num_init=num_initializations)

    def _check_shape(self, bsize, num_init):
        check_shape(self.image_ref, (-1, self.image_size, self.image_size))
        check_shape(self.keep_mask, (-1, self.image_size, self.image_size))
        check_shape(self.edt_ref_edge, (-1, self.image_size, self.image_size))
        check_shape(self.vertices, (-1, 3))
        check_shape(self.faces, (-1, 3))
        check_shape(self.rotations_matrix, (num_init, 3, 3))
        check_shape(self.translations, (num_init, 1, 3))
        check_shape(self.base_rotation, (bsize, 3, 3))
        check_shape(self.base_translation, (bsize, 1, 3))

    @property
    def rotations_matrix(self) -> torch.Tensor:
        """ (N, 3, 3) where N = num_initializations """
        rot_mats = rot6d_to_matrix(self.rotations)
        return rot_mats

    def apply_transformation(self):
        """
        Applies current rotation and translation to vertices.

            V_out = (V_model @ R + T) @ R_base + T_base,
        which first apply transformation in hand-space,
        then transform hand-space to camera-space.

        Out shape: (B, N_init, V, 3)
        """
        rots = self.rotations_matrix.unsqueeze(0) @ self.base_rotation.unsqueeze(1)
        transl = torch.add(
                torch.matmul(
                    self.translations.unsqueeze(0),  # (N, 1, 3) -> (1, N, 1, 3)
                    self.base_rotation.unsqueeze(1),  # (B, 3, 3) -> (B, 1, 3, 3)
                ),  # (B, N, 1, 3)
                self.base_translation.unsqueeze(1),  # (B, 1, 3) -> (B, 1, 1, 3)
            ) # (B, N, 1, 3)
        return torch.matmul(self.vertices, rots) + transl

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
        batch_K = self.renderer.K.unsqueeze(1).repeat(1, n, 1, 1)  # (B, N, 3, 3)
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

    def compute_edges(self, silhouette: torch.Tensor) -> torch.Tensor:
        return self.pool(silhouette) - silhouette

    def forward(self):
        """
        For losses, sum over dim=(2, 3) which are (H, W), 
        and take mean over dim=0 which is batch dimension.
        """
        verts = self.apply_transformation()
        image = self.keep_mask[:, None] * self.render()  # (B, N, W, W)
        image_ref = self.image_ref[:, None]
        b, n, w = image.size(0), image.size(1), self.image_size
        loss_dict = {}
        loss_dict["mask"] = torch.sum((image - image_ref)**2, dim=(2, 3)).mean(dim=0)
        with torch.no_grad():
            iou = ioumetrics.batch_mask_iou(
                image.view(b*n, w, w).detach(),
                image_ref.repeat(1, n, 1, 1).view(b*n, w, w).detach())  # (B*N,)
            iou = iou.view(b, n).mean(0)  # (N,)
        loss_dict["chamfer"] = self.lw_chamfer * torch.sum(
            self.compute_edges(image) * self.edt_ref_edge[:, None], 
            dim=(2, 3)).mean(dim=0)
        loss_dict["offscreen"] = 100000 * self.compute_offscreen_loss(verts)
        return loss_dict, iou, image

    @cached_property
    def fitted_results(self):
        """ At test-time, one should call self.fitted_results
        instead of self.forward() to get sorted version of
        (verts, loss, iou, image)
        """
        loss_dict, iou, _ = self.forward()
        inds = torch.argsort(iou, descending=True)
        for k, v in loss_dict.items():
            loss_dict[k] = v[inds]
        iou = iou[inds]
        with torch.no_grad():
            verts = self.apply_transformation()[:, inds]
        rotations = self.rotations_matrix[inds]
        translations = self.translations[inds]
        return FitResult(verts, rotations, translations, loss_dict, iou)

    def clustered_results(self, K):
        _attr = f'_clustered_results_{K}'
        if hasattr(self, _attr):
            return getattr(self, _attr)
        verts_orig = self.vertices
        fitted_results = self.fitted_results
        rots = fitted_results.rotations
        # If with base, rots is rotation in hands space,
        # since they are applied to all B base poses, it's safe to compute just with rots
        distance_matrix = compute_pairwise_dist(verts_orig, rots, verbose=True)
        center_indices, _ = cluster_distance_matrix(distance_matrix, K=K)
        def indexing(obj, l):
            if isinstance(obj, dict):
                obj_out = dict()
                for k, v in obj.items():
                    obj_out[k] = v[l]
                return obj_out
            return obj[l]

        verts = fitted_results.verts[:, center_indices]
        rotations = fitted_results.rotations[center_indices]
        translations = fitted_results.translations[center_indices]
        loss_dict = indexing(fitted_results.loss_dict, center_indices)
        iou = fitted_results.iou[center_indices]
        res = FitResult(
            verts, rotations, translations, loss_dict, iou)
        setattr(self, _attr, res)
        return res

    def render(self) -> torch.Tensor:
        """
        Renders objects according to current rotation and translation.

        Returns:
            images: ndarray (B, N_init, W, W)
        """
        verts = self.apply_transformation()  # (B, N, V, 3)
        b = verts.size(0)
        n = verts.size(1)
        batch_faces = self.faces.repeat(b, n, 1, 1)
        batch_K = self.renderer.K.unsqueeze(1).repeat(1, n, 1, 1)  # (B, 3, 3) -> (B, N, 3, 3)
        images = self.renderer(
            verts.view(b*n, -1, 3),
            batch_faces.view(b*n, -1, 3),
            K=batch_K.view(b*n, -1, 3),
            mode='silhouettes')
        images = images.view(b, n, self.image_size, self.image_size)
        return images

    def render_np(self) -> np.ndarray:
        """
        Renders objects according to current rotation and translation.

        Returns:
            images: ndarray (B, N_init, W, W)
        """
        return self.render().detach().cpu().numpy()
