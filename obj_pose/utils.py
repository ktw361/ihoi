import numpy as np
from sklearn.neighbors import KDTree
import tqdm


def compute_pairwise_dist(verts_orig: np.ndarray,
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