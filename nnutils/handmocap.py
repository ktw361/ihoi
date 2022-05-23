"""
Wrapper for Hand Pose Estimator using HandMocap.
Wrapper stolen from https://github.com/facebookresearch/phosa/blob/15f864d68ed3ed4536f019ad5713dda388d7c666/phosa/bodymocap.py
See: https://github.com/facebookresearch/frankmocap
"""
import os
import os.path as osp
import numpy as np
import torch

from handmocap.hand_mocap_api import HandMocap
from handmocap.hand_bbox_detector import HandBboxDetector

from torchvision.transforms import ToTensor
from nnutils import image_utils, geom_utils

from nnutils.hand_utils import ManopthWrapper

def get_handmocap_predictor(
        mocap_dir='externals/frankmocap',
        checkpoint_hand='extra_data/hand_module/pretrained_weights/pose_shape_best.pth', 
        smpl_dir='extra_data/smpl/',
    ):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    hand_mocap = HandMocap(osp.join(mocap_dir, checkpoint_hand), 
        osp.join(mocap_dir, smpl_dir), device = device)
    return hand_mocap



def process_mocap_predictions(mocap_predictions, image, hand_wrapper=None, mask=None):
    """

    Note ToTensor() will convert mask value 255 to 1

    Args:
        mocap_predictions (dict): 
            - left_hand/right_hand dict
                pred_vertices_smpl (778, 3)
                pred_joints_smpl (21, 3)
                faces (1538, 3)
                bbox_scale_ratio ()
                bbox_top_left (2,)
                pred_camera (3,)
                img_cropped (224, 224, 3)
                pred_hand_pose (1, 48)
                pred_hand_betas (1, 10)
                pred_vertices_img (778, 3)
                pred_joints_img (21, 3)
        image (ndarray): (H, W, 3)
        hand_wrapper : ManopthWrapper()
        mask (ndarray): (H, W)

    Returns:
        data = 
            'cTh':  Tensor (1, 4, 4), from hand to camera ?
            'hA':   Tensor (1, 45), Mano Pose
            'image': Tensor (1, 3, 224, 224),
            'obj_mask': ndarray (H, W),
            'cam_f': Tensor (2,), [fx, fx]
            'cam_p': Tensor (2,), [0, 0]
        }
    """
    if hand_wrapper is None:
        hand_wrapper = ManopthWrapper().to('cpu')
    one_hand = mocap_predictions[0]['right_hand']

    pose = torch.FloatTensor(one_hand['pred_hand_pose'])
    rot, hA = pose[..., :3], pose[..., 3:]
    hA = hA + hand_wrapper.hand_mean

    # obj_bbox = image_utils.mask_to_bbox(mask, 'minmax')
    x1, y1 = one_hand['bbox_top_left'] 
    bbox_len = 224 / one_hand['bbox_scale_ratio']
    x2, y2 = x1 + bbox_len, y1 + bbox_len
    
    hand_bbox = np.array([x1,y1, x2, y2])
    hoi_bbox = image_utils.joint_bbox(hand_bbox)
    hoi_bbox = image_utils.square_bbox(hoi_bbox, 1)
    
    cTh, cam_f, cam_p = get_camera(one_hand['pred_camera'], one_hand['bbox_top_left'], one_hand['bbox_scale_ratio'], hoi_bbox, hand_wrapper, hA, rot)
    crop = image_utils.crop_resize(image, hoi_bbox, return_np=False)
    crop = ToTensor()(crop)[None] * 2 - 1  # Tensor(1, 3, 224, 224)

    if mask is None:
        mask = torch.ones([1, 1, crop.shape[-2], crop.shape[-1]])
    else:
        mask = image_utils.crop_resize(mask, hoi_bbox, return_np=False)
        mask = ToTensor()(mask)[None]
        print(mask.shape)


    data = {
        'cTh': geom_utils.matrix_to_se3(cTh),
        'hA': hA,
        'image': crop,
        'obj_mask': mask,
        'cam_f': cam_f,
        'cam_p': cam_p
    }
    return data


def get_handmocap_detector(view_type='ego_centric'):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    bbox_detector =  HandBboxDetector(view_type, device)
    return bbox_detector


def get_camera(pred_cam, hand_bbox_tl, bbox_scale, bbox, hand_wrapper, hA, rot, fx=10):
    """ 
    Args:
        pred_cam    (ndarray):  (3,)
        hand_bbox_tl(ndarray):  (2,), int64, hand box top and left
        bbox_scale  (float):    scalar 
        bbox        (ndarray):  float32, (4,)
        hand_wrapper         :  ManopthWrapper
        hA     (torch.Tensor):  Mano Pose (1, 45)
        rot    (torch.Tensor):  Mano Rotation (1, 3)
    
    Returns:
        cTh: from hand to camera space (?)
        f: focal length (1, 2) 
        p: (1, 2) zeros
    """
    new_center = (bbox[0:2] + bbox[2:4]) / 2
    new_size = max(bbox[2:4] - bbox[0:2])
    cam, topleft, scale = image_utils.crop_weak_cam(
        pred_cam, hand_bbox_tl, bbox_scale, new_center, new_size)
    s, tx, ty = cam
    
    f = torch.FloatTensor([[fx, fx]])
    p = torch.FloatTensor([[0, 0]])

    translate = torch.FloatTensor([[tx, ty, fx/s]])
    
    _, joints = hand_wrapper(
        geom_utils.matrix_to_se3(geom_utils.axis_angle_t_to_matrix(rot)), # (1,3)->(1,4,4)-> (1,12)
        hA)
    
    print(translate.shape, joints.shape)
    cTh = geom_utils.axis_angle_t_to_matrix(
        rot, translate - joints[:, 5])
    return cTh, f, p

