from typing import NamedTuple
import numpy as np
import torch
import torch.nn as nn
import neural_renderer as nr
from scipy.ndimage.morphology import distance_transform_edt

from obj_pose.utils import compute_pairwise_dist
from obj_pose.cluster_distance_matrix import cluster_distance_matrix

from libyana.metrics import iou as ioumetrics

from homan.utils.geometry import rot6d_to_matrix


REND_SIZE = 256


class FitResult(NamedTuple):
    # All content are already sorted by iou
    verts: torch.Tensor
    rotations: torch.Tensor
    translations: torch.Tensor
    loss_dict: dict
    iou: torch.Tensor
    image: torch.Tensor


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
            base_translation:  (B, 3)
            K: (B, 3, 3)
                local camera of the object

        """
        assert ref_image.shape[-1] == ref_image.shape[-2], "Must be square."
        super().__init__()
        dtype = vertices.dtype
        device = vertices.device

        vertices = torch.as_tensor(vertices, device=device)
        faces = torch.as_tensor(faces, device=device)
        textures = torch.ones(faces.shape[0], 1, 1, 1, 3,
                            dtype=torch.float32, device=device)
        self.register_buffer("vertices", vertices)
        self.register_buffer("faces", faces.repeat(num_initializations, 1, 1))
        self.register_buffer("textures", textures)
        if base_rotation is None:
            base_rotation = torch.eye(
                3, dtype=dtype, device=device).unsqueeze_(0)
        else:
            base_rotation = base_rotation.clone()
        if base_translation is None:
            base_translation = torch.zeros([1, 3], dtype=dtype, device=device)
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
        self.register_buffer(
            "edt_ref_edge", torch.from_numpy(edt))
        # Setup renderer.
        if camera_K is None:
            camera_K = torch.FloatTensor([
                [[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]]]).to(device)
        rot = torch.eye(3).unsqueeze(0).to(device)
        trans = torch.zeros(1, 3).to(device)
        self.renderer = nr.renderer.Renderer(
            image_size=ref_image.shape[-1],
            K=camera_K,
            R=rot,  # eye(3)
            t=trans,  # zero
            orig_size=1,
            anti_aliasing=False,
        )
        self.lw_chamfer = lw_chamfer
        self.K = camera_K

        self.to(device)

    @property
    def rotations_matrix(self) -> torch.Tensor:
        """ (num_initialization, 3, 3) """
        rot_mats = rot6d_to_matrix(self.rotations)
        return rot_mats

    def apply_transformation(self):
        """
        Applies current rotation and translation to vertices.

        V_out = (V_model @ R_base + T_base) @ R + T

        Out shape: (N_init, V, 3)
        """
        rots = self.base_rotation @ self.rotations_matrix
        transl = self.base_translation @ self.rotations_matrix + self.translations
        return torch.matmul(self.vertices, rots) + transl

    def compute_offscreen_loss(self, verts):
        """
        Computes loss for offscreen penalty. This is used to prevent the degenerate
        solution of moving the object offscreen to minimize the chamfer loss.
        """
        # On-screen means coord_xy between [-1, 1] and far > depth > 0
        proj = nr.projection(
            verts,
            self.renderer.K,
            self.renderer.R,
            self.renderer.t,
            self.renderer.dist_coeffs,
            orig_size=1,
        )
        coord_xy, coord_z = proj[:, :, :2], proj[:, :, 2:]
        zeros = torch.zeros_like(coord_z)
        lower_right = torch.max(coord_xy - 1,
                                zeros).sum(dim=(1, 2))  # Amount greater than 1
        upper_left = torch.max(-1 - coord_xy,
                               zeros).sum(dim=(1, 2))  # Amount less than -1
        behind = torch.max(-coord_z, zeros).sum(dim=(1, 2))
        too_far = torch.max(coord_z - self.renderer.far, zeros).sum(dim=(1, 2))
        return lower_right + upper_left + behind + too_far

    def compute_edges(self, silhouette: torch.Tensor) -> torch.Tensor:
        return self.pool(silhouette) - silhouette

    def forward(self):
        verts = self.apply_transformation()
        image = self.keep_mask * self.renderer(
            verts, self.faces, mode="silhouettes")
        loss_dict = {}
        loss_dict["mask"] = torch.sum((image - self.image_ref)**2, dim=(1, 2))
        with torch.no_grad():
            iou = ioumetrics.batch_mask_iou(image.detach(),
                                            self.image_ref.detach())
        loss_dict["chamfer"] = self.lw_chamfer * torch.sum(
            self.compute_edges(image) * self.edt_ref_edge, dim=(1, 2))
        loss_dict["offscreen"] = 100000 * self.compute_offscreen_loss(verts)
        return loss_dict, iou, image
    
    # @cached_property
    @property
    def fitted_results(self):
        """ At test-time, one should call self.fitted_results
        instead of self.forward() to get sorted version of 
        (verts, loss, iou, image)
        """
        loss_dict, iou, image = self.forward()
        inds = torch.argsort(iou, descending=True)
        for k, v in loss_dict.items():
            loss_dict[k] = v[inds]
        iou = iou[inds]
        image = image[inds]
        with torch.no_grad():
            verts = self.apply_transformation()[inds]
        rotations = self.rotations_matrix[inds]
        translations = self.translations[inds]
        return FitResult(verts, rotations, translations, loss_dict, iou, image)

    def clustered_results(self, K):
        _attr = f'_clustered_results_{K}'
        if hasattr(self, _attr):
            return getattr(self, _attr)
        verts_orig = self.vertices
        fitted_results = self.fitted_results
        rots = fitted_results.rotations
        distance_matrix = compute_pairwise_dist(verts_orig, rots, verbose=True)
        center_indices, clusters = cluster_distance_matrix(distance_matrix, K=K)
        def indexing(obj, l):
            if isinstance(obj, dict):
                obj_out = dict()
                for k, v in obj.items():
                    obj_out[k] = v[l]
                return obj_out
            return obj[l]
            
        res = FitResult(*tuple(
            indexing(getattr(fitted_results, k), center_indices) 
            for k in fitted_results._fields))
        setattr(self, _attr, res)
        return res

    def render(self) -> np.ndarray:
        """
        Renders objects according to current rotation and translation.

        Returns:
            images: ndarray (N_init, W, W)
        """
        verts = self.apply_transformation()
        images = self.renderer(verts, self.faces, mode='silhouettes')
        images = images.detach().cpu().numpy()
        return images
