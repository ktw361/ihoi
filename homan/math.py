import torch
from pytorch3d.transforms import (
    matrix_to_quaternion, quaternion_to_matrix,
    rotation_6d_to_matrix, matrix_to_rotation_6d,
)


""" Average quaternions 
# https://math.stackexchange.com/questions/61146/averaging-quaternions
"""
def avg_rot6d_approx(rot6d: torch.Tensor, weights=None) -> torch.Tensor:
    """
    Args:
        rot6d: (N, 6)
    Returns:
        rot6dAvg: (6,)
    """
    matrices = rotation_6d_to_matrix(rot6d)
    matrixAvg = avg_matrix_approx(matrices, weights)
    return matrix_to_rotation_6d(matrixAvg)

def avg_matrix_approx(matrices: torch.Tensor, weights=None) -> torch.Tensor:
    """
    Args:
        matrices: (N, 3, 3)
    Returns:
        matrixAvg: (3, 3)
    """
    quats = matrix_to_quaternion(matrices)
    qAvg = avg_quaternions_approx(quats, weights)
    return quaternion_to_matrix(qAvg)
    
def avg_quaternions_approx(quats: torch.Tensor, weights=None) -> torch.Tensor:
    """
    Args:
        quats: (N, 4)
    Returns:
        qAvg: (4,)
    """
    if weights is not None and len(quats) != len(weights):
        raise ValueError("Args are of different length")
    if weights is None:
        weights = torch.ones_like(quats[:, 0])
    qAvg = torch.zeros_like(quats[0])
    for i, q in enumerate(quats):
        # Correct for double cover, by ensuring that dot product
        # of quats[i] and quats[0] is positive
        if i > 0 and torch.dot(quats[i], quats[0]) < 0.0:
            weights[i] = -weights[i]
        qAvg += weights[i] * q
    return qAvg / torch.norm(qAvg)
    

def avg_quaternions_eigen(quats: torch.Tensor, weights=None) -> torch.Tensor:
    if weights is not None and len(quats) != len(weights):
        raise ValueError("Args are of different length")
    if weights is None:
        weights = torch.ones_like(quats[:, 0])
    accum = torch.zeros((4, 4), device=quats.device)
    for i, q in enumerate(quats):
        qOuterProdWeighted = torch.outer(q, q) * weights[i]
        accum += qOuterProdWeighted
    _, eigVecs = torch.symeig(accum, eigenvectors=True)
    return eigVecs[:, 0]