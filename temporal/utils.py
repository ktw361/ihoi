from typing import Tuple
import torch
import numpy as np

from homan.utils.geometry import compute_random_rotations
from homan.lib3d.optitrans import TCO_init_from_boxes_zup_autodepth

from libzhifan.numeric import check_shape


def init_6d_pose_from_bboxes(bboxes: torch.Tensor, 
                             verts: torch.Tensor, 
                             cam_mat: torch.Tensor,
                             num_init: int,
                             base_rotation=None,
                             base_translation=None) -> Tuple[torch.Tensor]:
    """
    Args:
        bboxes: (T, 4)
        verts: (V, 3)
        cam_mat: (T, 3, 3)
        base_rotation: (T, 3, 3)
        base_translation: (T, 1, 3)

    Returns:
        rotations: (num_init, 3, 3)
        translations: (num_init, 1, 3)
    """
    device = 'cuda'
    bsize = len(bboxes)
    rotations = compute_random_rotations(B=num_init, upright=False, device=device)
    check_shape(cam_mat, (bsize, 3, 3))
    if base_rotation is not None:
        check_shape(base_rotation, (bsize, 3, 3))

    if base_rotation is None:
        base_rotation = torch.eye(
            3, dtype=rotations.dtype, device=device).unsqueeze_(0)
        base_rotation = base_rotation.repeat(bsize, 1, 1)
    else:
        base_rotation = base_rotation.clone()
    if base_translation is None:
        base_translation = torch.zeros(
            [bsize, 1, 3], dtype=rotations.dtype, device=device)
    else:
        base_translation = base_translation.clone()

    """ V_out = (V_model @ R + T) @ R_base + T_base """
    V_rotated = torch.matmul(
        (verts @ rotations).unsqueeze_(0),
        base_rotation.unsqueeze(1))  # (B, N_init, V, 3)

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
