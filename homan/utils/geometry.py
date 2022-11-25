# Copyright (c) Facebook, Inc. and its affiliates.
import math

import torch
from torch.nn import functional as F


def rot6d_to_matrix(rot_6d):
    """
    Convert 6D rotation representation to 3x3 rotation matrix.
    Reference: Zhou et al., "On the Continuity of Rotation Representations in Neural
    Networks", CVPR 2019

    Args:
        rot_6d (B x 6): Batch of 6D Rotation representation.

    Returns:
        Rotation matrices (B x 3 x 3).
    """
    rot_6d = rot_6d.view(-1, 3, 2)
    a1 = rot_6d[:, :, 0]
    a2 = rot_6d[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


def matrix_to_rot6d(rotmat):
    """
    Convert rotation matrix to 6D rotation representation.

    Args:
        rotmat (B x 3 x 3): Batch of rotation matrices.

    Returns:
        6D Rotations (B x 3 x 2).
    """
    return rotmat.view(-1, 3, 3)[:, :, :2]


def combine_verts(verts_list):
    batch_size = verts_list[0].shape[0]
    all_verts_list = [v.reshape(batch_size, -1, 3) for v in verts_list]
    verts_combined = torch.cat(all_verts_list, 1)
    return verts_combined


def center_vertices(vertices, faces, flip_y=True):
    """
    Centroid-align vertices.

    Args:
        vertices (V x 3): Vertices.
        faces (F x 3): Faces.
        flip_y (bool): If True, flips y verts to keep with image coordinates convention.

    Returns:
        vertices, faces
    """
    vertices = vertices - vertices.mean(dim=0, keepdim=True)
    if flip_y:
        vertices[:, 1] *= -1
        faces = faces[:, [2, 1, 0]]
    return vertices, faces


def compute_dist_z(verts1, verts2):
    """
    Computes distance between sets of vertices only in Z-direction.

    Args:
        verts1 (V x 3).
        verts2 (V x 3).

    Returns:
        tensor
    """
    a = verts1[:, 2].min()
    b = verts1[:, 2].max()
    c = verts2[:, 2].min()
    d = verts2[:, 2].max()
    if d >= a and b >= c:
        return 0.0
    return torch.min(torch.abs(c - b), torch.abs(a - d))


def compute_random_rotations(B=10, upright=False, device='cuda'):
    """
    Randomly samples rotation matrices.

    Args:
        B (int): Batch size.
        upright (bool): If True, samples rotations that are mostly upright. Otherwise,
            samples uniformly from rotation space.

    Returns:
        rotation_matrices (B x 3 x 3).
    """
    if upright:
        raise NotImplementedError
        a1 = torch.FloatTensor(B, 1).uniform_(0, 2 * math.pi)
        a2 = torch.FloatTensor(B, 1).uniform_(-math.pi / 6, math.pi / 6)
        a3 = torch.FloatTensor(B, 1).uniform_(-math.pi / 12, math.pi / 12)

        angles = torch.cat((a1, a2, a3), 1).cuda()
        rotation_matrices = euler_angles_to_matrix(angles, "YXZ")
    else:
        # Reference: J Avro. "Fast Random Rotation Matrices." (1992)
        x1, x2, x3 = torch.split(torch.rand(3 * B, device=device), B)
        tau = 2 * math.pi
        R = torch.stack(
            (  # B x 3 x 3
                torch.stack((torch.cos(tau * x1), torch.sin(
                    tau * x1), torch.zeros_like(x1)), 1),
                torch.stack((-torch.sin(tau * x1), torch.cos(
                    tau * x1), torch.zeros_like(x1)), 1),
                torch.stack((torch.zeros_like(x1), torch.zeros_like(x1),
                             torch.ones_like(x1)), 1),
            ),
            1,
        )
        v = torch.stack(
            (  # B x 3
                torch.cos(tau * x2) * torch.sqrt(x3),
                torch.sin(tau * x2) * torch.sqrt(x3),
                torch.sqrt(1 - x3),
            ),
            1,
        )
        identity = torch.eye(3, device=device).repeat(B, 1, 1)
        H = identity - 2 * v.unsqueeze(2) * v.unsqueeze(1)
        rotation_matrices = -torch.matmul(H, R)
    return rotation_matrices


def grid_rotations_spiral(num_sphere_pts: int, num_xy_rots: int) -> torch.Tensor:
    """ Deterministic algorithm.
    
    First distribute num_sphere_pts points, set z-axis of these points to the center;
    Then from each point on the sphere, divide the xy rotation into num_xy_rots, i.e. each of 2*pi/num_xy_rots angles.
    
    [Spiral Ref]: http://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/
    
    Return: 
        (num_sphere_pts * num_xy_rots, 3, 3) 
    """
    n = num_sphere_pts
    if n >= 600000:
        epsilon = 214
    elif n>= 400000:
        epsilon = 75
    elif n>= 11000:
        epsilon = 27
    elif n>= 890:
        epsilon = 10
    elif n>= 177:
        epsilon = 3.33
    elif n>= 24:
        epsilon = 1.33
    else:
        epsilon = 0.33

    goldenRatio = (1 + 5**0.5)/2
    i = torch.arange(0, n) 
    theta = 2 * math.pi * i / goldenRatio
    phi = torch.acos(1 - 2*(i+epsilon)/(n-1+2*epsilon))
    x, y, z = torch.cos(theta) * torch.sin(phi), torch.sin(theta) * torch.sin(phi), torch.cos(phi)
    z_vecs = torch.stack([x, y, z], 1)
    
    up_vecs = torch.zeros_like(z_vecs)
    up_vecs[:, 1] = 1.0  # (x=0, y=1, z=0)
    y_vecs = torch.nn.functional.normalize(torch.cross(z_vecs, up_vecs), p=2, dim=1)  # (num_sphere_pts, 3)
    x_vecs = torch.nn.functional.normalize(torch.cross(y_vecs, z_vecs), p=2, dim=1)
    Rz = torch.stack([x_vecs, y_vecs, z_vecs], dim=2)
    
    rads = 2*math.pi / num_xy_rots * torch.arange(num_xy_rots)
    Rxy = torch.stack([
        torch.stack([torch.cos(rads), torch.sin(rads), torch.zeros_like(rads)], 1),
        torch.stack([-torch.sin(rads), torch.cos(rads), torch.zeros_like(rads)], 1),
        torch.stack([torch.zeros_like(rads), torch.zeros_like(rads), torch.ones_like(rads)], 1),
    ], 1)  # (num_xy_rots, 3, 3)
    
    num_rots = num_sphere_pts * num_xy_rots
    Rxy = Rxy.unsqueeze(1).tile(1, num_sphere_pts, 1, 1).view(num_rots, 3, 3)
    Rz = Rz.unsqueeze(0).tile(num_xy_rots, 1, 1, 1).view(num_rots, 3, 3)
    rot_mats = Rz.matmul(Rxy)
    return rot_mats