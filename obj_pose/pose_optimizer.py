""" 
Refinement based pose optimizer.

Origin: homan/pose_optimization.py

"""

import os
from typing import NamedTuple

from tqdm import tqdm
import trimesh
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import neural_renderer as nr
from scipy.ndimage.morphology import distance_transform_edt

import torchvision.transforms.functional as F
from torchvision import transforms
from pytorch3d.transforms import Transform3d

from nnutils import image_utils, geom_utils
from nnutils.hand_utils import ManopthWrapper

from libzhifan.geometry import (
    SimpleMesh, CameraManager, projection
)
from libzhifan.odlib import xyxy_to_xywh, xywh_to_xyxy

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


class MeshHolder(NamedTuple):
    vertices: torch.Tensor
    faces: torch.Tensor
    

REND_SIZE = 256


class PoseRenderer(nn.Module):
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
        device='cuda',
    ):
        assert ref_image.shape[0] == ref_image.shape[1], "Must be square."
        super().__init__()

        vertices = torch.as_tensor(vertices, device=device)
        faces = torch.as_tensor(faces, device=device)
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


class PoseOptimization:

    OBJ_SCALES = dict(
        plate=0.3,
        cup=0.12,
        can=0.2,
        mug=0.12,
        bowl=0.12,
        bottle=0.2
    )

    WEAK_CAM_FX = 10

    FULL_HEIGHT = 720
    FULL_WIDTH = 1280

    """ 
    Bounding boxes mentioned in the pipeline:

    `hand_bbox`, `obj_bbox`: original boxes labels in epichoa 
    `hand_bbox_proc`: processed hand_bbox during mocap regressing
    `obj_bbox_squared`: squared obj_bbox before diff rendering

    """

    def __init__(self, 
                 obj_models_root='./weights/obj_models'):
        self.obj_models_root = obj_models_root

        self.obj_models_cache = dict()
    
    def load_obj_by_name(self, name, return_mesh=False, normalize=True):
        """ 
        Args:
            return_mesh: 
                if True, return an SimpleMesh object that ca

        Returns:
            a Mesh Object that has attributes:
                `vertices` and `faces`
        """
        if name not in self.obj_models_cache:
            obj_path = os.path.join(self.obj_models_root, f"{name}.obj")
            obj_scale = self.OBJ_SCALES[name]
            obj = trimesh.load(obj_path, force='mesh')
            verts = np.float32(obj.vertices)
            verts = verts - verts.mean(0)
            verts = verts / np.linalg.norm(verts, 2, 1).max() * obj_scale / 2
            obj_mesh = MeshHolder(vertices=verts, faces=obj.faces)
            self.obj_models_cache[name] = obj_mesh
        
        obj_mesh = self.obj_models_cache[name]
        
        if return_mesh:
            obj_mesh = SimpleMesh(obj_mesh.vertices, obj_mesh.faces)
        
        return obj_mesh
            
    def calc_hand_mesh(self, 
                       mocap_predictions,
                       return_mesh=False):
        """ wrapper of self._calc_hand_mesh.
        Given mocap_prediction (a list of dict), compute right_hand mesh.

        Args:
            return_mesh: if True, returns a SimpleMesh as output hand_mesh
        """
        one_hand = mocap_predictions[0]['right_hand']
        return self._calc_hand_mesh(one_hand, return_mesh=return_mesh)

    def _calc_hand_mesh(self, 
                        one_hand, 
                        return_mesh=False,
                        ):
        """ 

        When Ihoi predict the MANO params, 
        it takes the `hand_bbox_list` from dataset,
        then pad_resize the bbox, which results in the `bbox_processed`.

        The MANO params are predicted in 
            global_cam.crop(hand_bbox).resize(224, 224),
        so we need to emulate this process, see CameraManager below.

        Args:
            one_hand: dict
                - 'pred_hand_pose'
                - 'pred_hand_betas': Unused

                - 'pred_camera': used for translate hand_mesh to convert hand_mesh
                    so that result in a weak perspective camera. 

                - 'bbox_processed': bounding box for hand
                    this box should be used in mocap_predictor
            
            return_mesh: if True, returns a SimpleMesh as output hand_mesh
        
        Returns:
            hand_mesh: (1, V, 3) torch.Tensor
            hand_camera: CameraManager
            hand_bbox_proc: (4,) hand bounding box XYWH in original image
                same as one_hand['bbox_processed']
            global_camera: CameraManager
        """
        hand_wrapper = ManopthWrapper(flat_hand_mean=False).to('cpu')

        pred_hand_pose, pred_hand_betas, pred_camera = map(
            lambda x: torch.as_tensor(one_hand[x]),
            ('pred_hand_pose', 'pred_hand_betas', 'pred_camera'))
        rot_axisang, pred_hand_pose = pred_hand_pose[:, :3], pred_hand_pose[:, 3:]

        glb_rot = geom_utils.matrix_to_se3(
            geom_utils.axis_angle_t_to_matrix(rot_axisang)) # (1,3)->(1,4,4)-> (1,12)
        v, _, _, _ = hand_wrapper(None, pred_hand_pose, mode='inner', return_mesh=False)
        _, joints = hand_wrapper(
            glb_rot,
            pred_hand_pose, return_mesh=True)
        fx = self.WEAK_CAM_FX
        s, tx, ty = pred_camera
        translate = torch.FloatTensor([[tx, ty, fx/s]])
        cTh = geom_utils.axis_angle_t_to_matrix(
            rot_axisang, translate - joints[:, 5])
        cTh = Transform3d(matrix=cTh.transpose(1, 2))
        v = cTh.transform_points(v)

        out_mesh = v
        if return_mesh:
            out_mesh = SimpleMesh(v, hand_wrapper.hand_faces, copy_orig=True)
        hand_bbox_proc = one_hand['bbox_processed']
        hand_cam = self._hand_cam_from_bbox(hand_bbox_proc)
        hand_h, hand_w = hand_bbox_proc[2:]
        hand_cam = self._hand_cam_from_bbox(hand_bbox_proc)
        global_cam = hand_cam.resize(hand_h, hand_w).uncrop(
            hand_bbox_proc, self.FULL_HEIGHT, self.FULL_WIDTH
        )
        return out_mesh, hand_cam, hand_bbox_proc, global_cam
    
    def _hand_cam_from_bbox(self, hand_bbox):
        """ 
        Args:
            hand_bbox: (4,) in global screen space
                possibly hand_bbox processed after mocap
        """
        hand_crop_h = hand_crop_w = 224
        fx = self.WEAK_CAM_FX
        hand_cam = CameraManager(
            fx=fx, fy=fx, cx=0, cy=0, img_h=hand_bbox[2], img_w=hand_bbox[3],
            in_ndc=True
        ).resize(hand_crop_h, hand_crop_w)
        return hand_cam

    def render_model_output(self, 
                            mocap_predictions, 
                            fit_model, 
                            idx: int,
                            kind: str,
                            image,
                            obj_bbox=None,
                            ):
        """ 
        Args:
            idx: index into model.apply_transformation()
            kind: str, one of 
                - 'global': render w.r.t full image
                - 'ihoi': render w.r.t image prepared 
                    according to process_mocap_predictions()
            image: original full image
        """
        hand_mesh, hand_camera, hand_bbox_proc, global_cam = \
            self.calc_hand_mesh(mocap_predictions, return_mesh=True)
        obj_mesh = SimpleMesh(fit_model.apply_transformation()[idx], fit_model.faces[idx])
        if kind == 'global':
            img = projection.perspective_projection_by_camera(
                [obj_mesh, hand_mesh],
                global_cam,
                method=dict(
                    name='pytorch3d',
                    coor_sys='nr',
                    in_ndc=False
                ),
                image=image
            )
            return img
        elif kind == 'ihoi':
            obj_bbox_squared, image_patch, _ = \
                self._get_bbox_and_crop(image, image, obj_bbox,
                                        obj_bbox_expand=0.5, rend_size=REND_SIZE)
            hand_cam_expand = global_cam.crop_and_resize(
                obj_bbox_squared, REND_SIZE)
            img = projection.perspective_projection_by_camera(
                [obj_mesh, hand_mesh],
                hand_cam_expand,
                method=dict(
                    name='pytorch3d',
                    coor_sys='nr',
                    in_ndc=False
                ),
                image=np.uint8(image_patch*256),
            )
            return img
        else:
            raise ValueError(f"kind {kind} not unserstood")

    @staticmethod
    def pad_and_crop(image, box, out_size: int):
        """ Pad 0's if box exceeds boundary.

        Args:
            image: (H, W) or (H, W, 3)
            box: (4,) xywh
            out_size: int
        Returns:
            img_crop: (crop_h, crop_w, ...) according to input
        """
        x, y, w, h = box
        pad_x, pad_y = map(
            lambda t: int(np.ceil(max(-t, 0))), (x, y))
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Pad([pad_x, pad_y])
            ])
        x += pad_x
        y += pad_y

        image_pad = transform(image)
        crop_tensor = F.resized_crop(
            torch.as_tensor(image_pad)[None],
            int(y), int(x), int(h), int(w), size=[out_size, out_size],
            interpolation=transforms.InterpolationMode.NEAREST
        )
        return crop_tensor.permute(0, 2, 3, 1).squeeze().numpy()
    
    def _get_bbox_and_crop(self, 
                           image, 
                           object_mask, 
                           obj_bbox, 
                           rend_size=REND_SIZE,
                           obj_bbox_expand=0.5):
        """ 
        Returns:
            obj_box_squared
            image_patch: (H, W, 3) in [0, 1]
            object_mask_patch: (H, W) in [0, 1]

        """

        """ Get `obj_bbox` and `obj_bbox_sqaured` both in XYWH"""
        # x, y, w, h = obj_bbox
        # obj_bbox_xyxy = np.asarray([x, y, x + w, y + h], dtype=np.float32)
        obj_bbox_xyxy = xywh_to_xyxy(obj_bbox)
        obj_bbox_squared_xyxy = image_utils.square_bbox(obj_bbox_xyxy, obj_bbox_expand)
        # _x1, _y1, _x2, _y2 = obj_bbox_squared_xyxy
        # obj_bbox_squared = np.asarray([_x1, _y1, _x2 - _x1, _y2 - _y1], dtype=np.float32)
        obj_bbox_squared = xyxy_to_xywh(obj_bbox_squared_xyxy)
        # _x1, _y1, _w, _h = obj_bbox_squared

        """ Get `image` and `mask` """
        # obj_patch = image_utils.crop_resize(
        #     image, obj_bbox_squared_xyxy, final_size=rend_size)
        image_patch = self.pad_and_crop( 
            image, obj_bbox_squared, rend_size)
        obj_mask_patch = self.pad_and_crop(
            object_mask, obj_bbox_squared, rend_size)
        # obj_mask_patch = F.resized_crop(
        #     torch.as_tensor(object_mask)[None],
        #     int(_y1), int(_x1), int(_h), int(_w), size=[rend_size, rend_size],
        #     interpolation=transforms.InterpolationMode.NEAREST
        # )[0].numpy()

        return obj_bbox_squared, image_patch, obj_mask_patch

    def optimize_obj_pose(self,
                          image,
                          obj_bbox,
                          object_mask,
                          cat,
                          hand_bbox_processed,
                          global_cam=None,
                          return_kwargs=False,

                          num_initializations=400,
                          num_iterations=50,
                          debug=True,
                          sort_best=False,
                          viz=True,
                          ):
        """ 
        Args: See EpicInference dataset output.
            image: (H, W, 3) torch.Tensor. possibly (720, 1280, 3)
            obj_bbox: (4,)
            object_mask: (H, W) int ndarray of (-1, 0, 1)
            cat: str
            hand_bbox:processed: (4,)
                Note this differs from `hand_bbox` directly from dataset
            
        Returns:
            PoseRenderer
        """
        rend_size = REND_SIZE

        obj_bbox_squared, image_patch, obj_mask_patch = \
            self._get_bbox_and_crop(image, object_mask, obj_bbox,
                                    obj_bbox_expand=0.5, rend_size=rend_size)

        """ Get global_camera. """
        if global_cam is None:
            hand_h, hand_w = hand_bbox_processed[2:]
            hand_cam = self._hand_cam_from_bbox(hand_bbox_processed)
            global_cam = hand_cam.resize(hand_h, hand_w).uncrop(
                hand_bbox_processed, self.FULL_HEIGHT, self.FULL_WIDTH
            )

        obj_mesh = self.load_obj_by_name(cat, return_mesh=False)
        vertices = torch.as_tensor(obj_mesh.vertices, device='cuda')
        faces = torch.as_tensor(obj_mesh.faces, device='cuda')

        if return_kwargs:
            return dict(
                vertices=vertices,
                faces=faces,
                bbox=obj_bbox,
                square_bbox=obj_bbox_squared,
                image=image_patch,
                mask=obj_mask_patch,
                K=global_cam.get_K(),
                image_size=(rend_size, rend_size),

                num_initializations=num_initializations,
                num_iterations=num_iterations,
                debug=debug,
                sort_best=sort_best,
                viz=viz,

                global_cam=global_cam,
            )
            
        else:
            model = find_optimal_pose(
                vertices=vertices,
                faces=faces,
                bbox=obj_bbox,
                square_bbox=obj_bbox_squared,
                image=image_patch,
                mask=obj_mask_patch,
                K=global_cam.get_K(),
                image_size=(rend_size, rend_size),

                num_initializations=num_initializations,
                num_iterations=num_iterations,
                debug=debug,
                sort_best=sort_best,
                viz=viz,
            )
            return model


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
        vertices: torch.Tensor
        faces: torch.Tensor
        mask: 1 for fg, 0 for bg, -1 for ignored
        bbox: XYWH, bbox from original data source. E.g. epichoa
        square_bbox: XYWH, Enlarged and squared from `bbox`
            The box that matches `mask`
        image_size:
        K: (3, 3) ndarray, represents the global camera that
            captures the original image.
        
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
    os.makedirs(viz_folder, exist_ok=True)
    device = vertices.device
    ts = 1
    textures = torch.ones(faces.shape[0], ts, ts, ts, 3,
                          dtype=torch.float32, device=device)
    x, y, b, _ = square_bbox
    L = max(image_size[:2])
    camintr_roi = kcrop.get_K_crop_resize(
        torch.Tensor(K).unsqueeze(0), torch.tensor([[x, y, x + b, y + b]]),
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
    K = npt.tensorify(K).unsqueeze(0).to(vertices.device)
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
    translations_init = compute_optimal_translation(
        bbox_target=np.array(bbox) * REND_SIZE / L,
        vertices=torch.matmul(vertices.unsqueeze(0), rotations_init),
        f=K[0, 0, 0].item() / max(image_size))
    translations_init = TCO_init_from_boxes_zup_autodepth(
        bbox, torch.matmul(vertices.unsqueeze(0), rotations_init),
        K).unsqueeze(1)
    if debug:
        # Debug shows initalized verts on image & mask
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

    model = PoseRenderer(
        ref_image=mask,
        vertices=vertices,
        faces=faces,
        textures=textures,
        rotation_init=matrix_to_rot6d(rotations_init),
        translation_init=translations_init,
        num_initializations=num_initializations,
        K=camintr_roi,
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
                sil, overlays=[mask,]*len(sil), viz_nb=4, 
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
