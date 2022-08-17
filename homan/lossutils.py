from typing import List
import numpy as np
import torch

from homan.interactions import contactloss, scenesdf

from libyana.visutils import imagify

# MANO_CLOSED_FACES = np.array(
#     trimesh.load("extra_data/mano/closed_fmano.obj", process=False).faces)
MANO_CLOSED_FACES = np.load("homan/local_data/closed_fmano.npy")


def compute_smooth_loss(verts_hand,
                        verts_obj):
    raise NotImplementedError


def compute_pca_loss(mano_pca_comps):
    return {"pca_mean": (mano_pca_comps**2).mean()}


def compute_collision_loss(verts_hand,
                           verts_object,
                           faces_hand,
                           faces_object,
                           max_collisions=5000,
                           debug=True):
    hand_nb = verts_hand.shape[0] // verts_object.shape[0]
    mano_faces = faces_object[0].new(MANO_CLOSED_FACES)
    if hand_nb > 1:
        mano_faces = faces_object[0].new(MANO_CLOSED_FACES[:, ::-1].copy())
        sdfl = scenesdf.SDFSceneLoss(
            [mano_faces, mano_faces, faces_object[0]])
        hand_verts = [
            verts_hand[hand_idx::2] for hand_idx in range(hand_nb)
        ]
        sdf_loss, sdf_meta = sdfl(hand_verts + [verts_object])
    else:
        sdfl = scenesdf.SDFSceneLoss([mano_faces, faces_object[0]])
        sdf_loss, sdf_meta = sdfl([verts_hand, verts_object])
    return {"collision": sdf_loss.mean()}


def compute_intrinsic_scale_prior(intrinsic_scales, intrinsic_mean):
    return torch.sum(
        (intrinsic_scales - intrinsic_mean)**2) / intrinsic_scales.shape[0]


def compute_contact_loss(verts_hand_b, verts_object_b, faces_object,
                         faces_hand):
    hand_nb = verts_hand_b.shape[0] // verts_object_b.shape[0]
    faces_hand_closed = faces_hand.new(MANO_CLOSED_FACES).unsqueeze(0)
    if hand_nb > 1:
        missed_losses = []
        contact_losses = []
        for hand_idx in range(hand_nb):
            hand_verts = verts_hand_b[hand_idx::hand_nb]
            missed_loss, contact_loss, _, _ = contactloss.compute_contact_loss(
                hand_verts, faces_hand_closed, verts_object_b, faces_object)
            missed_losses.append(missed_loss)
            contact_losses.append(contact_loss)
        missed_loss = torch.stack(missed_losses).mean()
        contact_loss = torch.stack(contact_losses).mean()
    else:
        missed_loss, contact_loss, _, _ = contactloss.compute_contact_loss(
            verts_hand_b, faces_hand_closed, verts_object_b, faces_object)
    return {"contact": missed_loss + contact_loss}, None


def compute_ordinal_depth_loss(masks:torch.Tensor, 
                               silhouettes: List[torch.Tensor], 
                               depths: List[torch.Tensor],
                               ):
    """
    Args:
        masks: (B, obj_nb, height, width), ground truth mask (spongy)
        silhouettes: [(B, height, width), ...] of len obj_nb, Rendered silhouettes
        depths: [(B, height, width), ...] of len obj_nb, Rendered depths
    """
    loss = torch.as_tensor(0.0).float().cuda()
    num_pairs = 1e-7
    # Create square mask to match square renders
    height = masks.shape[2]
    width = masks.shape[3]
    silhouettes = [silh[:, :height, :width] for silh in silhouettes]
    depths = [depth[:, :height, :width] for depth in depths]
    for i in range(len(silhouettes)):
        for j in range(len(silhouettes)):
            if i == j:
                continue
            has_pred = silhouettes[i] & silhouettes[j]
            pairs = (has_pred.sum([1, 2]) > 0).sum().item()
            if pairs == 0:
                continue
            num_pairs += pairs
            # front_i_gt: pixels which should be i (i closer), 
            #   also exclude those being rendered as background as input mask 
            #   may not align with rendered silhoueets perfectly
            front_i_gt = masks[:, i] & (~masks[:, j])
            front_j_pred = depths[j] < depths[i]  # but get j closer
            mask = front_i_gt & front_j_pred & has_pred
            if mask.sum() == 0:
                continue
            dists = torch.clamp(depths[i] - depths[j], min=0.0, max=2.0)
            loss += torch.sum(torch.log(1 + torch.exp(dists))[mask])
    loss /= num_pairs
    return {"depth": loss}


def iou_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Args:
        pred: (..., H, W) values in [0, 1]
        gt: (..., H, W) values in [0, 1]
    
    Returns:
        loss: (...,) in [0, inf)
    """
    union = pred + gt
    inter = pred * gt
    iou = inter.sum(dim=(-2, -1)) / ((union - inter).sum(dim=(-2, -1)) + 1e-7)
    # loss = 1.0 - iou
    loss = - iou.log_()
    return loss


def rotation_loss_v1(Ra: torch.Tensor, Rb: torch.Tensor) -> torch.Tensor:
    """ 
    If Ra = [Ra_x; Ra_y; Ra_z],
    compute:
        cos_x = <Ra_x, Rb_x>
        loss_x = - ln((1+cos_x)/2)
        loss = loss_x + loss_y + loss_z
        
    Args:
        Ra, Rb: (B, 3, 3)
    Returns:
        loss: (B,) in [0, inf)
    """
    prod = torch.matmul(Ra.permute(0, 2, 1), Rb)
    cos_vec = prod[..., [0, 1, 2], [0, 1, 2]]
    loss = -1.0 * ((cos_vec + 1.0) / 2).log_()
    return loss.sum(1)