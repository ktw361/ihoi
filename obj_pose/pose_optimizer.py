""" 
Refinement based pose optimizer.

Origin: homan/pose_optimization.py

"""

import os

import matplotlib.pyplot as plt
import neural_renderer as nr
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from libyana.camutils import project
from libyana.conversions import npt
from libyana.lib3d import kcrop
from libyana.metrics import iou as ioumetrics
from libyana.visutils import imagify

from .lib3d.optitrans import (
    TCO_init_from_boxes_zup_autodepth,
    compute_optimal_translation,
)
from .utils.geometry import (
    compute_random_rotations,
    rot6d_to_matrix,
    matrix_to_rot6d,
)

REND_SIZE = 256

class PoseOptimizer(nn.Module):
    """
    Computes the optimal object pose from an instance mask and an exemplar mesh.
    Closely following PHOSA, we optimize an occlusion-aware silhouette loss
    that consists of a one-way chamfer loss and a silhouette matching loss.
    """
    def __init__(
        self,
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
    ):
        assert ref_image.shape[0] == ref_image.shape[1], "Must be square."
        super().__init__()

        self.register_buffer("vertices",
                             vertices.repeat(num_initializations, 1, 1))
        self.register_buffer("faces", faces.repeat(num_initializations, 1, 1))
        self.register_buffer(
            "textures", textures.repeat(num_initializations, 1, 1, 1, 1, 1))

        # Load reference mask.
        # Convention for silhouette-aware loss: -1=occlusion, 0=bg, 1=fg.
        image_ref = torch.from_numpy((ref_image > 0).astype(np.float32))
        keep_mask = torch.from_numpy((ref_image >= 0).astype(np.float32))
        self.register_buffer("image_ref",
                             image_ref.repeat(num_initializations, 1, 1))
        self.register_buffer("keep_mask",
                             keep_mask.repeat(num_initializations, 1, 1))
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
            "edt_ref_edge",
            torch.from_numpy(edt).repeat(num_initializations, 1, 1).float())
        # Setup renderer.
        if K is None:
            K = torch.cuda.FloatTensor([[[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]]])
        rot = torch.eye(3).unsqueeze(0).cuda()
        trans = torch.zeros(1, 3).cuda()
        self.renderer = nr.renderer.Renderer(
            image_size=ref_image.shape[0],
            K=K,
            R=rot,
            t=trans,
            orig_size=1,
            anti_aliasing=False,
        )
        self.lw_chamfer = lw_chamfer
        self.K = K

    def apply_transformation(self):
        """
        Applies current rotation and translation to vertices.
        """
        rots = rot6d_to_matrix(self.rotations)
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

    def render(self):
        """
        Renders objects according to current rotation and translation.
        """
        verts = self.apply_transformation()
        images = self.renderer(verts, self.faces, torch.tanh(self.textures))[0]
        images = images.detach().cpu().numpy().transpose(0, 2, 3, 1)
        return images


def find_optimal_pose(
    vertices,
    faces,
    mask,
    bbox,
    square_bbox,
    image_size,
    K,
    num_iterations=50,
    num_initializations=2000,
    lr=1e-2,
    image=None,
    debug=True,
    viz_folder="output/tmp",
    viz_step=10,
    sort_best=True,
    rotations_init=None,
    viz=True,
):
    """
    Args:
        vertices:
        faces:
        mask: 1 for fg, 0 for bg, -1 for ignored
        bbox: XYWH, bbox from original data source. E.g. epichoa
        square_bbox: XYWH, Enlarged and squared from `bbox`
        image_size:
        K: (3, 3) ndarray
        

    Returns:
        A PoseOptimizer Object,
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
    os.makedirs(viz_folder, exist_ok=True)
    ts = 1
    textures = torch.ones(faces.shape[0], ts, ts, ts, 3,
                          dtype=torch.float32).cuda()
    x, y, b, _ = square_bbox
    L = max(image_size[:2])
    camintr_roi = kcrop.get_K_crop_resize(
        torch.Tensor(K).unsqueeze(0), torch.tensor([[x, y, x + b, y + b]]),
        [REND_SIZE]).cuda()
    # Stuff to keep around
    best_losses = torch.tensor(np.inf)
    best_rots = None
    best_trans = None
    best_loss_single = torch.tensor(np.inf)
    best_rots_single = None
    best_trans_single = None
    loop = tqdm(total=num_iterations)
    K = npt.tensorify(K).unsqueeze(0).to(vertices.device)
    # Mask is in format 256 x 256 (REND_SIZE x REND_SIZE)
    # bbox is in xywh in original image space
    # K is in pixel space
    # If initial rotation is not provided, it is sampled
    # uniformly from SO3
    if rotations_init is None:
        rotations_init = compute_random_rotations(num_initializations,
                                                  upright=False)

    # Translation is retrieved by matching the tight bbox of projected
    # vertices with the bbox of the target mask
    translations_init = compute_optimal_translation(
        bbox_target=np.array(bbox) * REND_SIZE / L,
        vertices=torch.matmul(vertices.unsqueeze(0), rotations_init),
        f=K[0, 0, 0].item() / max(image_size))
    translations_init = TCO_init_from_boxes_zup_autodepth(
        bbox, torch.matmul(vertices.unsqueeze(0), rotations_init),
        K).unsqueeze(1)
    if debug:
        trans_verts = translations_init + torch.matmul(vertices,
                                                       rotations_init)
        proj_verts = project.batch_proj2d(trans_verts,
                                          K.repeat(trans_verts.shape[0], 1,
                                                   1)).cpu()
        verts3d = trans_verts.cpu()
        flat_verts = proj_verts.contiguous().view(-1, 2)
        if viz:
            plt.clf()
            fig, axes = plt.subplots(1, 3)
            ax = axes[0]
            ax.imshow(image)
            ax.scatter(flat_verts[:, 0], flat_verts[:, 1], s=1, alpha=0.2)
            ax = axes[1]
            ax.imshow(image)
            for vert in proj_verts:
                ax.scatter(vert[:, 0], vert[:, 1], s=1, alpha=0.2)
            ax = axes[2]
            for vert in verts3d:
                ax.scatter(vert[:, 0], vert[:, 2], s=1, alpha=0.2)

            fig.savefig(os.path.join(viz_folder, "autotrans.png"))
            plt.close()

        proj_verts = project.batch_proj2d(
            trans_verts, camintr_roi.repeat(trans_verts.shape[0], 1, 1)).cpu()
        flat_verts = proj_verts.contiguous().view(-1, 2)
        if viz:
            fig, axes = plt.subplots(1, 3)
            ax = axes[0]
            ax.imshow(mask)
            ax.scatter(flat_verts[:, 0], flat_verts[:, 1], s=1, alpha=0.2)
            ax = axes[1]
            ax.imshow(mask)
            for vert in proj_verts:
                ax.scatter(vert[:, 0], vert[:, 1], s=1, alpha=0.2)
            ax = axes[2]
            for vert in verts3d:
                ax.scatter(vert[:, 0], vert[:, 2], s=1, alpha=0.2)
            fig.savefig(os.path.join(viz_folder, "autotrans_roi.png"))
            plt.close()

    # Bring crop K to NC rendering space
    camintr_roi[:, :2] = camintr_roi[:, :2] / REND_SIZE

    model = PoseOptimizer(
        ref_image=mask,
        vertices=vertices,
        faces=faces,
        textures=textures,
        rotation_init=matrix_to_rot6d(rotations_init),
        translation_init=translations_init,
        num_initializations=num_initializations,
        K=camintr_roi,
    )
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(num_iterations):
        optimizer.zero_grad()
        loss_dict, iou, sil = model()
        if debug and (step % viz_step == 0):
            debug_viz_folder = os.path.join(viz_folder, "poseoptim")
            os.makedirs(debug_viz_folder, exist_ok=True)
            imagify.viz_imgrow(sil,
                               overlays=[
                                   mask,
                               ] * len(sil),
                               viz_nb=4,
                               path=os.path.join(debug_viz_folder,
                                                 f"{step:04d}.png"))

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


""" 
Removed:
def find_optimal_poses(image_size,
                       faces=None,
                       vertices=None,
                       annotations=None,
                       images=None,
                       Ks=None,
                       num_iterations=50,
                       num_initializations=2000,
                       viz_path="tmp.png",
                       debug=False):
"""