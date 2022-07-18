import torch
import numpy as np
from sklearn.neighbors import KDTree
from pytorch3d.ops import knn_points
import tqdm


def compute_pairwise_dist(verts_orig: torch.Tensor,
                          rot_mats: torch.Tensor,
                          verbose=True) -> torch.Tensor:
    """
    Args:
        verts: (V, 3) original vertices before rot and transl.
            We don't need translation
        rot_mats: (N, 3, 3) rotation matrix
            
    Returns:
        dist_matrix: (N, N) upper triangle matrix
    """
    N = len(rot_mats)
    res = torch.zeros([N, N], dtype=torch.float32)
    with torch.no_grad():
        verts = torch.matmul(verts_orig, rot_mats)
        loop = tqdm.trange(N, desc="Compute dist matrix") if verbose else range(N)
        for i in loop:
            verts_i = verts[[i]].repeat(N, 1, 1)
            dist = knn_points(
                verts, verts_i,
                K=1, return_nn=False, return_sorted=False
            ).dists
            res[i, :] = torch.mean(torch.sqrt(dist), dim=1).squeeze()
    return res


def compute_pairwise_dist_cpu(verts_orig: np.ndarray,
                              rot_mats: np.ndarray,
                              verbose=False) -> np.ndarray:
    """
    Args:
        verts: (V, 3) original vertices before rot and transl.
            We don't need translation
        rot_mats: (N, 3, 3) rotation matrix
            
    Returns:
        dist_matrix: (N, N) upper triangle matrix
    """
    N = len(rot_mats)
    verts = verts_orig @ rot_mats
    res = np.zeros([N, N], dtype=np.float32)
    loop = tqdm.trange(N, desc="Compute dist matrix") if verbose else range(N)
    for i in loop:
        kdt = KDTree(verts[i], metric='euclidean')
        for j in range(i+1, N):
            distance, _ = kdt.query(verts[j], k=1)
            res[i, j] = np.mean(distance)
    return res