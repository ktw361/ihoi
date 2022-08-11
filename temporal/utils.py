import torch
import numpy as np


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
