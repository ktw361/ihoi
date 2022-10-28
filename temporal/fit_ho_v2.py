import argparse
import tqdm
import numpy as np
import torch
from datetime import datetime
from torch.utils import tensorboard
from datasets.epic_clip import EpicClipDataset
from homan.ho_forwarder_v2 import HOForwarderV2Impl, HOForwarderV2Vis
from homan.utils.geometry import matrix_to_rot6d, rot6d_to_matrix
from nnutils.handmocap import (
    get_handmocap_predictor,
    collate_mocap_hand,
    recover_pca_pose, compute_hand_transform,
)

from obj_pose.obj_loader import OBJLoader

from nnutils import image_utils
from temporal.optim_plan import (
    optimize_hand, smooth_hand_pose)
from temporal.optim_plan import sampled_obj_optimize
from temporal.utils import init_6d_pose_from_bboxes, softmax_temp


def get_args():
    parser = argparse.ArgumentParser(
        description="Simple Epic inference")
    parser.add_argument(
        "index", type=int, default=5,
        help="Index into gt_clips.json")
    parser.add_argument(
        "--image_sets",
        default='/home/skynet/Zhifan/data/epic_analysis/gt_clips.json')

    args = parser.parse_args()
    return args


def main(args):
    dataset = EpicClipDataset(
        image_sets='/home/skynet/Zhifan/data/epic_analysis/gt_clips.json',
        sample_frames=30)

    obj_loader = OBJLoader()
    hand_predictor = get_handmocap_predictor()
    writer = tensorboard.writer.SummaryWriter(
        log_dir=f"runs/{datetime.now().strftime('%m-%d_%H:%M')}")

    if args.index >= 0:
        fit_scene(dataset, hand_predictor, obj_loader,
                  args.index, writer=None)
        writer.close()
        return

    for index in tqdm.trange(len(dataset)):
        fit_scene(dataset, hand_predictor, obj_loader,
                    index, writer=None)

    writer.close()


def fit_scene(dataset,
              hand_predictor,
              obj_loader,
              index: int,
              writer=None):
    device = 'cuda'
    info = dataset.data_infos[index]
    images, hand_bbox_dicts, side, obj_bboxes, hand_masks, obj_masks, cat = dataset[index]
    global_cam = dataset.get_camera(index)

    """ Process all hands """
    mocap_predictions = []
    for img, hand_dict in zip(images, hand_bbox_dicts):
        mocap_pred = hand_predictor.regress(
            img[..., ::-1], [hand_dict]
        )
        mocap_predictions += mocap_pred
    one_hands = collate_mocap_hand(mocap_predictions, side)

    """ Extract mocap_output """
    pred_hand_full_pose, pred_hand_betas, pred_camera = map(
        lambda x: torch.as_tensor(one_hands[x], device=device),
        ('pred_hand_pose', 'pred_hand_betas', 'pred_camera'))
    hand_bbox_proc = one_hands['bbox_processed']
    rot_axisang, pred_hand_pose = pred_hand_full_pose[:,
                                                      :3], pred_hand_full_pose[:, 3:]
    mano_pca_pose = recover_pca_pose(pred_hand_pose, side)

    hand_sz = torch.ones_like(global_cam.fx) * 224
    hand_cam = global_cam.crop(hand_bbox_proc).resize(new_w=hand_sz, new_h=hand_sz)
    hand_rotation, hand_translation = compute_hand_transform(
        rot_axisang, pred_hand_pose, pred_camera, side,
        hand_cam=hand_cam)

    """ Extract mask input """
    ihoi_box_expand = 0.3  # 1.0
    rend_size = 256  # TODO
    USE_HAND_BBOX = True
    hand_bboxes = torch.as_tensor(np.stack([v[side] for v in hand_bbox_dicts]))
    bbox_squared = image_utils.square_bbox_xywh(
        hand_bboxes if USE_HAND_BBOX else obj_bboxes, ihoi_box_expand).int()
    obj_mask_patch = image_utils.batch_crop_resize(
        obj_masks, bbox_squared, rend_size)
    hand_mask_patch = image_utils.batch_crop_resize(
        hand_masks, bbox_squared, rend_size)
    image_patch = image_utils.batch_crop_resize(
        images, bbox_squared, rend_size)
    ihoi_h = torch.ones([len(global_cam)]) * rend_size
    ihoi_w = torch.ones([len(global_cam)]) * rend_size
    ihoi_cam = global_cam.crop(bbox_squared).resize(ihoi_h, ihoi_w)

    homan = HOForwarderV2Vis(
        camintr=ihoi_cam.to_nr(return_mat=True),
        ihoi_img_patch=image_patch,
    )
    homan.set_hand_params(
        rotations_hand=hand_rotation,
        translations_hand=hand_translation,
        hand_side=side,
        mano_pca_pose=mano_pca_pose,
        mano_betas=pred_hand_betas)
    homan.set_hand_target(target_masks_hand=hand_mask_patch)
    """
    Step 1. Interpolate pca_pose
    Step 2. Optimize hand_mask
    """
    fmt = f'output/temporal/{info.vid}_{info.gt_frame}_%s'
    # homan.render_grid(obj_idx=-1, with_hand=False, low_reso=False).savefig(fmt % 'input')
    # homan.render_grid(obj_idx=-1, with_hand=True, low_reso=False).savefig(fmt % 'raw')
    print("Optimize hand")
    homan = smooth_hand_pose(homan, lr=0.1)
    homan = optimize_hand(homan, verbose=False)

    obj_mesh = obj_loader.load_obj_by_name(cat, return_mesh=False)
    vertices = torch.as_tensor(obj_mesh.vertices, device='cuda')
    faces = torch.as_tensor(obj_mesh.faces, device='cuda')
    num_initializations = 1
    K_global = global_cam.get_K()

    device = 'cuda'
    rotations, translations = init_6d_pose_from_bboxes(
        obj_bboxes, vertices, cam_mat=K_global,
        num_init=num_initializations,
        base_rotation=rot6d_to_matrix(homan.rotations_hand),
        base_translation=homan.translations_hand)

    homan.set_obj_params(
        translations_object=translations,
        rotations_object=rotations,
        verts_object_og=vertices,
        faces_object=faces,
        scale_object=1.0)
    homan.set_obj_target(obj_mask_patch)

    """
    Step 4. Optimize both hand+object mask using best object pose
    """
    # homan.info = info
    # homan = optimize_hand_allmask(
    #     homan, num_steps=250, vis_interval=10, writer=writer)

    temp = 10
    homan, weights = sampled_obj_optimize(
        homan, num_epochs=10, num_iters=200, vis_interval=35,
        temperature=temp,
        ratio=0.75)

    homan.render_grid(obj_idx=0, with_hand=True, low_reso=False).savefig(fmt % 'optim.png')
    homan.to_scene(show_axis=False).export((fmt % 'mesh.obj'))
    torch.save(homan, (fmt % 'model.pth'))


if __name__ == '__main__':
    main(get_args())
