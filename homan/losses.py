#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=import-error,no-member,wrong-import-order,too-many-arguments,too-many-instance-attributes
# pylint: disable=missing-function-docstring,missing-module-docstring,missing-class-docstring
import neural_renderer as nr
import torch

from homan.constants import INTERACTION_MAPPING, REND_SIZE
from homan.utils.bbox import compute_iou
from homan.utils.geometry import compute_dist_z

from libyana.camutils import project
from libyana.metrics.iou import batch_mask_iou

# from physim.datasets import collate


def project_bbox(vertices, renderer, bbox_expansion=0.0):
    """
    Computes the 2D bounding box of the vertices after projected to the image plane.

    Args:
        vertices (V x 3).
        renderer: Renderer used to get camera parameters.
        bbox_expansion (float): Amount to expand the bounding boxes.

    Returns:
        If a part_label dict is given, returns a dictionary mapping part name to bbox.
        Else, returns the projected 2D bounding box.
    """
    worldverts = (vertices * torch.Tensor([[[1, -1, 1.0]]]).cuda())
    proj = nr.projection(
        worldverts,
        K=renderer.K,
        R=renderer.R,
        t=renderer.t,
        dist_coeffs=renderer.dist_coeffs,
        orig_size=1,
    )
    proj = proj[:, :, :2]
    bboxes_xy = torch.cat([proj.min(1)[0], proj.max(1)[0]], 1)
    if bbox_expansion:
        center = (bboxes_xy[:, :2] + bboxes_xy[:, 2:]) / 2
        extent = (bboxes_xy[:, 2:] - bboxes_xy[:, :2]) / 2 * (1 +
                                                              bbox_expansion)
        bboxes_xy = torch.cat([center - extent, center + extent], 1)
    return bboxes_xy


def batch_pairwise_dist(x, y):
    """
    Args:
        x: (B, P, D)
    """
    bs, num_points_x, points_dim = x.size()
    _, num_points_y, _ = y.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind_x = torch.arange(0, num_points_x, device=x.device)
    diag_ind_y = torch.arange(0, num_points_y, device=x.device)
    rx = (
        xx[:, diag_ind_x, diag_ind_x]
        .unsqueeze(1)
        .expand_as(zz.transpose(2, 1))
    )
    ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
    P = rx.transpose(2, 1) + ry - 2 * zz
    return P


class Losses():
    def __init__(
        self,
        renderer,
        ref_mask_object,
        keep_mask_object,
        ref_mask_hand,
        keep_mask_hand,
        camintr_rois_object,
        camintr_rois_hand,
        camintr,
        class_name='default',
        inter_type="min",
        hand_nb=1,
    ):
        """
        Args:
            inter_type (str): [centroid|min] centroid to penalize centroid distances
                min to penalize minimum distance
        """
        self.renderer = nr.renderer.Renderer(image_size=REND_SIZE,
                                             K=renderer.K,
                                             R=renderer.R,
                                             t=renderer.t,
                                             orig_size=1)
        self.inter_type = inter_type
        self.ref_mask_object = ref_mask_object
        self.keep_mask_object = keep_mask_object
        self.camintr_rois_object = camintr_rois_object
        self.ref_mask_hand = ref_mask_hand
        self.keep_mask_hand = keep_mask_hand
        self.camintr_rois_hand = camintr_rois_hand
        # Necessary ! Otherwise camintr gets updated for some reason TODO check
        self.camintr = camintr.clone()
        self.thresh = 3  # z thresh for interaction loss
        self.mse = torch.nn.MSELoss()
        self.class_name = class_name
        self.hand_nb = hand_nb

        self.expansion = 0.2
        self.interaction_map = INTERACTION_MAPPING[class_name]

        self.interaction_pairs = None

    def assign_interaction_pairs(self, verts_hand, verts_object):
        """
        Assigns pairs of people and objects that are interacting.
        Unlike PHOSA, one obect can be assigned to multiple hands but one hand is
        only assigned to a given object (e.g. we assume that a hand is
        not manipulating two obects at the same time)

        This is computed separately from the loss function because there are potential
        speed improvements to re-using stale interaction pairs across multiple
        iterations (although not currently being done).

        A person and an object are interacting if the 3D bounding boxes overlap:
            * Check if X-Y bounding boxes overlap by projecting to image plane (with
              some expansion defined by BBOX_EXPANSION), and
            * Check if Z overlaps by thresholding distance.

        Args:
            verts_hand (B x V_p x 3).
            verts_object (B x V_o x 3).

        Returns:
            interaction_pairs: List[Tuple(person_index, object_index)]
        """
        interacting = []

        with torch.no_grad():
            bboxes_object = project_bbox(verts_object,
                                         self.renderer,
                                         bbox_expansion=self.expansion)
            bboxes_hand = project_bbox(verts_hand,
                                       self.renderer,
                                       bbox_expansion=self.expansion)
            for batch_idx, (box_obj, box_hand) in enumerate(
                    zip(bboxes_object, bboxes_hand)):
                iou = compute_iou(box_obj, box_hand)
                z_dist = compute_dist_z(verts_object[batch_idx],
                                        verts_hand[batch_idx])
                if (iou > 0) and (z_dist < self.thresh):
                    interacting.append(1)
                else:
                    interacting.append(0)
            return interacting

    def compute_sil_loss_hand(self, verts, faces):
        loss_sil = torch.Tensor([0.0]).float().cuda()
        for i in range(len(verts)):
            camintr = self.camintr_rois_hand[i]
            # Rendering happens in ROI
            rend = self.renderer(
                verts[[i], ...],
                faces[i],
                K=camintr.unsqueeze(0),
                mode="silhouettes")
            image = self.keep_mask_hand * rend
            l_m = torch.sum((image - self.ref_mask_hand)**
                            2) / self.keep_mask_hand.sum()
            loss_sil += l_m
        return {"loss_sil_hand": loss_sil / len(verts)}

    def compute_sil_loss_object(self, verts, faces):
        loss_sil = torch.Tensor([0.0]).float().cuda()
        # Rendering happens in ROI
        camintr = self.camintr_rois_object
        rend = self.renderer(verts, faces, K=camintr, mode="silhouettes")
        image = self.keep_mask_object * rend
        l_m = torch.sum(
            (image - self.ref_mask_object)**2) / self.keep_mask_object.sum()
        loss_sil += l_m
        ious = batch_mask_iou(image, self.ref_mask_object)
        return {
            "loss_sil_obj": loss_sil / len(verts)
        }, {
            'iou_object': ious.mean().item()
        }

    def compute_interaction_loss(self, verts_hand_b, verts_object_b):
        """
        Computes interaction loss.
        Args:
            verts_hand_b (B, person_nb, vert_nb, 3)
            verts_object_b (B, object_nb, vert_nb, 3)
        """
        loss_inter = torch.Tensor([0.0]).float().cuda()
        num_interactions = 0
        min_dists = []
        for person_idx in range(verts_hand_b.shape[1]):
            for object_idx in range(verts_object_b.shape[1]):
                interacting = self.assign_interaction_pairs(
                    verts_hand_b[:, person_idx], verts_object_b[:, object_idx])
                for batch_idx, inter in enumerate(interacting):
                    if inter:
                        v_p = verts_hand_b[batch_idx, person_idx]
                        v_o = verts_object_b[batch_idx, object_idx]
                        if self.inter_type == "centroid":
                            inter_error = self.mse(v_p.mean(0), v_o.mean(0))
                        elif self.inter_type == "min":
                            inter_error = batch_pairwise_dist(
                                v_p.unsqueeze(0), v_o.unsqueeze(0)).min()
                        loss_inter += inter_error
                        num_interactions += 1
                # Compute minimal vertex distance
                with torch.no_grad():
                    min_dist = torch.sqrt(
                        batch_pairwise_dist(
                            verts_hand_b[:, person_idx],
                            verts_object_b[:, object_idx])).min(1)[0].min(1)[0]
                min_dists.append(min_dist)

        # Avoid nans by 0 division
        if num_interactions > 0:
            loss_inter_ = loss_inter / num_interactions
        else:
            loss_inter = loss_inter
        min_dists = torch.stack(min_dists).min(0)[0]
        return {
            "loss_inter": loss_inter
        }, {
            "handobj_maxdist": torch.max(min_dists).item()
        }
