import numpy as np
import torch
import torch.nn as nn
import neural_renderer as nr
from scipy.ndimage.morphology import distance_transform_edt

from libyana.metrics import iou as ioumetrics

from homan.utils.geometry import rot6d_to_matrix


REND_SIZE = 256


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
                 textures,
                 rotation_init,
                 translation_init,
                 num_initializations=1,
                 kernel_size=7,
                 K=None,
                 power=0.25,
                 lw_chamfer=0,
                 device='cuda'):
        """ 
        Args:
            K: local camera of the object

        """
        assert ref_image.shape[0] == ref_image.shape[1], "Must be square."
        super().__init__()

        vertices = torch.as_tensor(vertices, device=device)
        faces = torch.as_tensor(faces, device=device)
        self.register_buffer("vertices", vertices)
        self.register_buffer("faces", faces.repeat(num_initializations, 1, 1))
        self.register_buffer("textures", textures)

        # Load reference mask.
        # Convention for silhouette-aware loss: -1=occlusion, 0=bg, 1=fg.
        image_ref = torch.from_numpy((ref_image > 0).astype(np.float32))
        keep_mask = torch.from_numpy((ref_image >= 0).astype(np.float32))
        self.register_buffer("image_ref", image_ref)
        self.register_buffer("keep_mask", keep_mask)
        self.pool = torch.nn.MaxPool2d(kernel_size=kernel_size,
                                       stride=1,
                                       padding=(kernel_size // 2))
        self.rotations = nn.Parameter(rotation_init.clone().float(),
                                      requires_grad=True)
        if rotation_init.shape[0] != translation_init.shape[0]:
            translation_init = translation_init.repeat(num_initializations, 1,
                                                       1)
        self.translations = nn.Parameter(translation_init.clone().float(),
                                         requires_grad=True)
        mask_edge = self.compute_edges(image_ref.unsqueeze(0)).cpu().numpy()
        edt = distance_transform_edt(1 - (mask_edge > 0))**(power * 2)
        self.register_buffer(
            "edt_ref_edge", torch.from_numpy(edt))
        # Setup renderer.
        if K is None:
            K = torch.FloatTensor([[[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]]]).to(device)
        rot = torch.eye(3).unsqueeze(0).to(device)
        trans = torch.zeros(1, 3).to(device)
        self.renderer = nr.renderer.Renderer(
            image_size=ref_image.shape[0],
            K=K,
            R=rot,  # eye(3)
            t=trans,  # zero
            orig_size=1,
            anti_aliasing=False,
        )
        self.lw_chamfer = lw_chamfer
        self.K = K

        self.to(device)

    @property
    def rotations_matrix(self) -> torch.Tensor:
        """ (num_initialization, 3, 3) """
        rot_mats = rot6d_to_matrix(self.rotations)
        return rot_mats

    def apply_transformation(self):
        """
        Applies current rotation and translation to vertices.
        """
        rots = self.rotations_matrix
        return torch.matmul(self.vertices, rots) + self.translations

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

    def compute_edges(self, silhouette):
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

    def render(self) -> np.ndarray:
        """
        Renders objects according to current rotation and translation.
        """
        verts = self.apply_transformation()
        # images = self.renderer(verts, self.faces, torch.tanh(self.textures))[0]
        images = self.renderer(verts, self.faces, mode='silhouettes')
        images = images.detach().cpu().numpy()
        return images
