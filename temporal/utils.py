from functools import reduce
from typing import Tuple
import torch
import numpy as np

from homan.ho_forwarder_v2 import HOForwarderV2Impl
from homan.utils.geometry import (
    compute_random_rotations, generate_rotations_o2h
)
from homan.lib3d.optitrans import TCO_init_from_boxes_zup_autodepth
from homan.math import avg_matrix_approx
from pytorch3d.transforms import matrix_to_rotation_6d

from libzhifan.numeric import check_shape


def get_base_transform(bsize, 
                       base_rotation=None, 
                       base_translation=None,
                       device='cuda',
                       dtype=torch.float32):
    """
    Returns:
        base_rotation: (B, 3, 3)
        base_translation: (B, 1, 3)
    """
    if base_rotation is None:
        base_rotation = torch.eye(
            3, dtype=dtype, device=device).unsqueeze_(0)
        base_rotation = base_rotation.repeat(bsize, 1, 1)
    else:
        base_rotation = base_rotation.clone()
    if base_translation is None:
        base_translation = torch.zeros(
            [bsize, 1, 3], dtype=dtype, device=device)
    else:
        base_translation = base_translation.clone()
    return base_rotation, base_translation


def init_6d_pose_from_bboxes(bboxes: torch.Tensor, 
                             verts: torch.Tensor, 
                             cam_mat: torch.Tensor,
                             num_init: int,
                             base_rotation=None,
                             base_translation=None,
                             zero_init_transl=False) -> Tuple[torch.Tensor]:
    """
    # TODO make this function conform to init_6d_pose_from_bboxes_v2
    Args:
        bboxes: (T, 4) xywh
        verts: (V, 3)
        cam_mat: (T, 3, 3)
        base_rotation: (T, 3, 3) apply to row-vec
        base_translation: (T, 1, 3)

    Returns:
        rotations: (num_init, 3, 3)
        translations: (num_init, 1, 3)
    """
    device = 'cuda'
    bsize = len(cam_mat)
    rotations = compute_random_rotations(B=num_init, upright=False, device=device)
    check_shape(cam_mat, (bsize, 3, 3))
    if base_rotation is not None:
        check_shape(base_rotation, (bsize, 3, 3))
    
    base_rotation, base_translation = get_base_transform(
        bsize,
        base_rotation=base_rotation, base_translation=base_translation,
        device=device, dtype=rotations.dtype)

    """ V_out = (V_model @ R + T) @ R_base + T_base """
    V_rotated = torch.matmul(
        (verts @ rotations).unsqueeze_(0),
        base_rotation.unsqueeze(1))  # (B, N_init, V, 3)

    if zero_init_transl:
        translations_init = rotations.new_zeros([num_init, 1, 3])
    else:
        translations_init = []
        for i in range(bsize):
            _translations_init = TCO_init_from_boxes_zup_autodepth(
                bboxes[[i],...], V_rotated[i], cam_mat[[i],...], 
                ).unsqueeze_(1)
            _translations_init -= base_translation[i]  # (N, 1, 3) - (1, 1, 3)
            _translations_init = _translations_init @ base_rotation[i].T  # inv
            translations_init.append(_translations_init)
        if cam_mat.size(0) == 1:
            translations_init = translations_init[0]
        else:
            translations_init = sum(translations_init) / len(translations_init)
    
    return rotations, translations_init


def init_6d_obj_pose_v2(global_bboxes: torch.Tensor,
                        verts_hand: torch.Tensor,
                        verts_obj: torch.Tensor,
                        global_cam_mat: torch.Tensor,
                        local_cam_mat: torch.Tensor,
                        rot_init: dict,
                        transl_init_method: str,
                        scale_init_method: str,
                        base_rotation=None,
                        base_translation=None,
                        homan: HOForwarderV2Impl = None):
    """ Compute random rotation, then estimate scale, then estimate z/depth, then xy
    Args:
        bboxes: (T, 4) xywh, T stands for Time 
        verts_hand: (T, V, 3)
        verts: (V, 3)
        global_cam_mat: (T, 3, 3) 
        local_cam_mat: (T, 3, 3) 
        rot_init: dict, see conf.yaml
        transl_init_method: one of {'zero', 'fingers', 'mask'}
        scale_init_method: one of {'one', 'xyz', 'est'}
            if 'est', will estimate using bboxes and hand size.
            note this assume homan.scale_mode != 'xyz'
        base_rotation: (T, 3, 3) can apply to row-vec
        base_translation: (T, 1, 3)
        homan: if transl_init_method == 'fingers', homan will be called.

    Returns: 
    Transformation from object to *base*, not to camera.
        rotations_6d: (num_init, 6)
        translations: (num_init, 1, 3) identical for all num_inits
        scale: identical for all num_inits
            (num_init, 3)   for 'xyz'
            (num_init,)     otherwise
    num_init = num_sphere_pts * num_xy_rots if rot_init.method == 'spiral' or 'upright'
    """
    device = 'cuda'
    bsize = len(local_cam_mat)
    """ rotations will be for col-vec """
    if rot_init['method'] == 'upright':
        avg_base_rotation = avg_matrix_approx(base_rotation).view(1, 3, 3)
        R_o2h, _ = generate_rotations_o2h(
            rot_init, base_rotations=avg_base_rotation, device=device)
    else:
        R_o2h, _ = generate_rotations_o2h(
            rot_init, base_rotations=None, device=device)
    num_init = R_o2h.size(0)
        
    check_shape(local_cam_mat, (bsize, 3, 3))
    base_rotation, base_translation = get_base_transform(
        bsize,
        base_rotation=base_rotation, base_translation=base_translation,
        device=device, dtype=R_o2h.dtype)

    """ V_out = (V_model @ R + T) @ R_base + T_base """
    R_o2h_row = R_o2h.permute(0, 2, 1)
    base_rot_row = base_rotation.permute(0, 2, 1)
    V_rotated = torch.matmul(
        (verts_obj @ R_o2h_row).unsqueeze_(0),
        base_rot_row.unsqueeze(1))  # (T, N_init, V, 3)

    if scale_init_method == 'one':
        scale_init = R_o2h.new_ones([num_init])
    elif scale_init_method == 'xyz':
        scale_init = R_o2h.new_ones([num_init, 3])
    elif scale_init_method == 'est':
        scale_init = estimate_obj_scale(global_bboxes, verts_hand, V_rotated, global_cam_mat)
        V_rotated = V_rotated * scale_init.view(1, -1, 1, 1)
    else:
        raise ValueError()

    if transl_init_method == 'zero':
        translations_init = R_o2h.new_zeros([num_init, 1, 3])
    elif transl_init_method == 'fingers':
        num_priors = 5
        v_hand = homan.get_verts_hand(hand_space=True)
        v_inds = reduce(lambda a, b: a + b, homan.contact_regions.verts[:num_priors], [])
        finger_center = v_hand[:, v_inds, :]  # (T, 3)
        translations_init = finger_center.mean(dim=(0, 1)).view(1, 1, 3).tile(num_init, 1, 1) 
    elif transl_init_method == 'mask':
        z_o2c = estimate_obj_depth(global_bboxes, V_rotated, local_cam_mat)  # (T, N_init), obj-to-cam
        x_o2c, y_o2c = estimate_obj_xy(global_bboxes, V_rotated, global_cam_mat, z_o2c)  # (T, N_init), obj-to-cam
        est_translations = torch.cat([x_o2c, y_o2c, z_o2c], dim=-1).view(bsize, num_init, 1, 3)
        translations_init = torch.einsum(
            'bnij,bkj->bnik', 
            est_translations - base_translation.unsqueeze(1), base_rotation)
        translations_init = translations_init.mean(dim=0)
    else:
        raise ValueError()
    
    R_o2h_6d = matrix_to_rotation_6d(R_o2h)
    return R_o2h_6d, translations_init, scale_init


def estimate_obj_scale(bboxes: torch.Tensor,
                       vh: torch.Tensor,
                       vo: torch.Tensor,
                       global_cam_mat: torch.Tensor,
                       debug: bool = False):
    """ Estimate scale of obj verts s.t.
        est_scale * Vo_3d / Vh_3d = box_diagonal / hand_proj_diagonal

    Args:
        bboxes: (T, 4) xywh, T stands for Time. Target boxes to match
        vh: (T, V, 3)
        vo: (T, N_init, V, 3) allow N_init verts
        cam_mat: (T, 3, 3)

    Returns:
        estimated_scale: (N_init,)
    """
    if debug:
        print(f"bboxes.shape: {bboxes.shape}")
        print(f"vh.shape: {vh.shape}")
        print(f"vo.shape: {vo.shape}")
        print(f"global_cam_mat.shape: {global_cam_mat.shape}")
    global_cam_mat = global_cam_mat.to(vo.device)
    bboxes = bboxes.to(vo.device)
    diag = (bboxes[:, 2]**2 + bboxes[:, 3]**2).sqrt()  # (T,)
    diag.unsqueeze_(1)

    # Get hand proj
    vh_proj = torch.einsum(
        'bvj,bij->bvi', vh, global_cam_mat)  # (T, V, 3)
    vh_xmin = vh_proj[..., 0].min(dim=1).values  # (T, 3)
    vh_xmax = vh_proj[..., 0].max(dim=1).values
    vh_ymin = vh_proj[..., 1].min(dim=1).values
    vh_ymax = vh_proj[..., 1].max(dim=1).values
    vh_diag = ((vh_xmax - vh_xmin)**2 + (vh_ymax - vh_ymin)**2).sqrt()
    vh_diag.unsqueeze_(1)

    vh_xmin = vh[..., 0].min(dim=1).values  # (T, 3)
    vh_xmax = vh[..., 0].max(dim=1).values
    vh_ymin = vh[..., 1].min(dim=1).values
    vh_ymax = vh[..., 1].max(dim=1).values
    vh_zmin = vh[..., 2].min(dim=1).values
    vh_zmax = vh[..., 2].max(dim=1).values
    vh_diameter = ((vh_xmax - vh_xmin)**2 + (vh_ymax - vh_ymin)**2 + (vh_zmax - vh_zmin)**2).sqrt()
    vh_diameter.unsqueeze_(1)

    vo_xmin = vo[..., 0].min(dim=2).values  # (T, N_init, 3)
    vo_xmax = vo[..., 0].max(dim=2).values
    vo_ymin = vo[..., 0].min(dim=2).values
    vo_ymax = vo[..., 0].max(dim=2).values
    vo_zmin = vo[..., 0].min(dim=2).values
    vo_zmax = vo[..., 0].max(dim=2).values
    vo_diameter = ((vo_xmax - vo_xmin)**2 + (vo_ymax - vo_ymin)**2 + (vo_zmax - vo_zmin)**2).sqrt()

    scale_est = (diag.log() - vh_diag.log() + vh_diameter.log() - vo_diameter.log()
                 ).mean(dim=0).exp()
    return scale_est


def estimate_obj_depth(bboxes: torch.Tensor,
                       vo: torch.Tensor,
                       local_cam_mat: torch.Tensor):
    """ Find depth/z s.t.
        sqrt( (fx*(X_diam/z))**2 + (fy*(Y_diam/z))**2 ) = bboxes_diag
    Note here z is object-to-camera.
    Further, to find best "z" so that V/z = diag,
        we use: exp{ Avg(log(V) - log(diag)) }

    Args:
        bboxes: (T, 4) xywh, T stands for Time. Target boxes to match
        vo: (T, N_init, V, 3) allow N_init verts
        cam_mat: (T, 3, 3)

    Returns:
        estimated_depth: (T, N_init,)
    """
    local_cam_mat = local_cam_mat.to(vo.device)
    bboxes = bboxes.to(vo.device)
    diag = (bboxes[:, 2]**2 + bboxes[:, 3]**2).sqrt().unsqueeze(1)  # (T, 1)

    fx = local_cam_mat[:, 0, 0].unsqueeze(1)
    fy = local_cam_mat[:, 1, 1].unsqueeze(1)
    vo_xmax_3d = vo[:, :, :, 0].max(dim=-1).values  # (T, N_init)
    vo_xmin_3d = vo[:, :, :, 0].min(dim=-1).values
    vo_ymax_3d = vo[:, :, :, 1].max(dim=-1).values
    vo_ymin_3d = vo[:, :, :, 1].min(dim=-1).values
    vo_proj_diameter = ((fx*(vo_xmax_3d - vo_xmin_3d))**2 + (fy*(vo_ymax_3d - vo_ymin_3d))**2).sqrt()

    # est_depth = (vo_proj_diameter.log() - diag.log()).mean(dim=0).exp()
    est_depth = vo_proj_diameter / diag
    return est_depth


def estimate_obj_xy(bboxes: torch.Tensor,
                    vo: torch.Tensor,
                    global_cam_mat: torch.Tensor,
                    z_o2c: torch.Tensor):
    """ Find x and y in obj-to-cam translation so that they match the boxes center.
        i.e:  1/Z*(Xc, Yc, Z) @ K = (xc, yc, 1)

    Args:
        bboxes: (T, 4) xywh, T stands for Time. Target boxes to match
        vo: (T, N_init, V, 3) allow N_init verts
        cam_mat: (T, 3, 3)
        z_o2c: (N_init,)

    Returns:
        estimated_x: (T, N_init,)
        estimated_y: (T, N_init,)
    """
    global_cam_mat = global_cam_mat.to(vo.device)
    bboxes = bboxes.to(vo.device)
    xc_box = (bboxes[:, 0] + bboxes[:, 2] / 2).unsqueeze(1)  # (N_init,)
    yc_box = (bboxes[:, 1] + bboxes[:, 3] / 2).unsqueeze(1)

    fx = global_cam_mat[:, 0, 0].unsqueeze(1)
    fy = global_cam_mat[:, 1, 1].unsqueeze(1)
    cx = global_cam_mat[:, 0, 2].unsqueeze(1)
    cy = global_cam_mat[:, 1, 2].unsqueeze(1)
    # xc_3d = ((xc_box - cx) / fx * z_o2c).mean(dim=0)
    # yc_3d = ((yc_box - cy) / fy * z_o2c).mean(dim=0)
    xc_3d = (xc_box - cx) / fx * z_o2c
    yc_3d = (yc_box - cy) / fy * z_o2c
    return xc_3d, yc_3d


def jitter_box(box: np.ndarray, ratio: float, num: int) -> np.ndarray:
    """
    Args:
        box: (4,) xywh
        ratio: in (0, 1)
            the change of box center and size roughly within this ratio
        num: number of output
    
    Returns:
        jittered_boxes: (num, 4)
    """
    x0, y0, w, h = box
    xc = x0 + w/2
    yc = y0 + w/2
    dx = w * ratio
    dy = h * ratio
    noise = np.random.randn(num, 4) * [dx, dy, dx, dy]
    boxes = np.ones([num, 4]) * [xc, yc, w, h]
    boxes = boxes + noise
    boxes[:, :2] = np.maximum(boxes[:, :2] - boxes[:, 2:]/2, 0.0)
    return boxes


def median_vector_index(vecs: torch.Tensor):
    """ Given a list of vector,
    find the vector that best describe this list.

    Args:
        vecs (torch.Tensor): (B, D)
        method: one of {'median', 'mean'}
    
    Returns:
        vec (torch.Tensor): (D,)
    """
    # if method == 'mean':
    #     return torch.mean(vecs, dim=0)
    # elif method == 'median':
    mean_vec = torch.mean(vecs, dim=0)
    dist = torch.sum((vecs - mean_vec)**2, dim=1)
    ind = torch.argmin(dist)
    return ind


def softmax_temp(weights: torch.Tensor, temperature) -> torch.Tensor:
    """ softmax with temperature 

    Args:
        weigths: (N,)
        temperature: float
    
    Returns:
        probs: (N,)
    """
    probs = torch.nn.functional.softmax(weights / temperature)
    return probs


def choose_with_softmax(weights: torch.Tensor, 
                        temperature: float, 
                        ratio: float) -> list:
    """ softmax with temperature 

    Args:
        weigths: (N,)
        temperature (float):
        ratio (float): in (0, 1]
    
    Returns:
        list of index
    """
    num_samples = int(len(weights) * ratio)
    with torch.no_grad():
        probs = softmax_temp(weights, temperature)
        indices = torch.multinomial(
            probs, num_samples=num_samples, replacement=False)
    return indices.tolist()