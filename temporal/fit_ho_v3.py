import hydra
from omegaconf import DictConfig, OmegaConf
import tqdm
import numpy as np
import torch
import logging
from moviepy import editor
import matplotlib.pyplot as plt

from datasets.epic_clip import EpicClipDataset
from datasets.epic_clip_v3 import EpicClipDatasetV3
from obj_pose.obj_loader import OBJLoader
from homan.ho_forwarder_v2 import HOForwarderV2Vis
from nnutils.handmocap import (
    get_handmocap_predictor, extract_forwarder_input
)
from temporal.optim_plan import (
    optimize_hand, smooth_hand_pose, reinit_sample_optimize
)
from temporal.utils import init_6d_obj_pose_v2
from temporal.visualize import make_compare_video

from libzhifan import io


@hydra.main(config_path='../config', config_name='conf')
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    log = logging.getLogger(__name__)

    assert cfg.optim_method != 'multiview', "Please fit_mvho.py"
    sample_frames = cfg.dataset.sample_frames
    if cfg.dataset.version == 'v2':
        dataset = EpicClipDataset(
            image_sets=cfg.dataset.image_sets,
            sample_frames=sample_frames)
    elif cfg.dataset.version == 'v3':
        dataset = EpicClipDatasetV3(
            image_sets=cfg.dataset.image_sets,
            sample_frames=sample_frames,
            show_loading_time=True)

    obj_loader = OBJLoader()
    hand_predictor = get_handmocap_predictor()

    if cfg.debug_index is not None:
        fit_scene(dataset, hand_predictor, obj_loader, cfg.debug_index, cfg=cfg)
        return

    for index in tqdm.trange(cfg.index_from, min(cfg.index_to, len(dataset))):
        try:
            fit_scene(dataset, hand_predictor, obj_loader, index, cfg=cfg)
            log.info(f"Succeed at index [{index}]: {dataset.data_infos[index]}")
        except Exception as e:
            log.info(f"Failed at index [{index}]: {dataset.data_infos[index]}. Reason: {e}")
            continue


def fit_scene(dataset,
              hand_predictor,
              obj_loader: OBJLoader,
              index: int,
              cfg: DictConfig = None):
    """
    Args:
        cfg: see config/conf.yaml
    """
    device = 'cuda'
    info = dataset.data_infos[index]
    data_elem = dataset[index]

    images, hand_bbox_dicts, side, obj_bboxes, \
        hand_masks, obj_masks, cat, global_cam = data_elem

    ihoi_cam_nr_mat, ihoi_cam_mat, image_patch, \
        hand_rotation_6d, hand_translation, \
        mano_pca_pose, pred_hand_betas, hand_mask_patch, obj_mask_patch = \
        extract_forwarder_input(
            data_elem, ihoi_box_expand=cfg.preprocess.ihoi_box_expand,
            device=device)

    homan = HOForwarderV2Vis(
        camintr=ihoi_cam_nr_mat,
        ihoi_img_patch=image_patch,
    )
    homan.set_hand_params(
        rotations_hand=hand_rotation_6d,
        translations_hand=hand_translation,
        hand_side=side,
        mano_pca_pose=mano_pca_pose,
        mano_betas=pred_hand_betas)
    homan.set_hand_target(target_masks_hand=hand_mask_patch)
    """
    Step 1. Interpolate pca_pose
    Step 2. Optimize hand_mask
    """
    if cfg.dataset.version == 'v2':
        fmt = f'{info.vid}_{info.gt_frame}_%s'
    elif cfg.dataset.version == 'v3':
        fmt = f'{info.vid}_{info.start}_{info.end}_%s'

    print("Smooth hand")
    homan = smooth_hand_pose(homan, lr=0.1)
    if cfg.homan.optimize_hand:
        print("Optimize hand")
        homan = optimize_hand(homan, verbose=False)

    obj_mesh = obj_loader.load_obj_by_name(cat, return_mesh=False)
    vertices = torch.as_tensor(obj_mesh.vertices, device='cuda')
    faces = torch.as_tensor(obj_mesh.faces, device='cuda')

    with torch.no_grad():
        rot_init = cfg.homan.rot_init[cat]
        rotation6d_inits, translation_inits, scale_inits = init_6d_obj_pose_v2(
            obj_bboxes, homan.get_verts_hand(), vertices,
            global_cam_mat=global_cam.get_K(), local_cam_mat=ihoi_cam_mat,
            rot_init=rot_init,
            transl_init_method=cfg.homan.transl_init_method,
            scale_init_method=cfg.homan.scale_init_method,
            base_rotation=homan.rot_mat_hand,
            base_translation=homan.translations_hand,
            homan=homan)

    homan.set_obj_params(
        translations_object=translation_inits,
        rotations_object=rotation6d_inits,
        verts_object_og=vertices,
        faces_object=faces,
        scale_mode=cfg.homan.scale_mode,
        scale_init=scale_inits)
    homan.set_obj_target(obj_mask_patch)
    if cfg.optim.obj_part_prior:
        part_v, _ = obj_loader.load_part_by_name(cat)
        homan.set_obj_part(part_verts=part_v)

    homan.render_grid(obj_idx=0, with_hand=False,
                      low_reso=False, overlay_gt=True).savefig(fmt % 'input.png')

    """
    Step 4. Optimize both hand+object mask using best object pose
    """
    save_grid = (fmt % 'optim.mp4') if cfg.save_optim_video else None
    homan, weights, results, best_metric = reinit_sample_optimize(
        homan, rotation6d_inits, translation_inits, scale_inits,
        save_grid=save_grid,
        cfg=cfg.optim)

    homan.to_scene(show_axis=False).export((fmt % 'mesh.obj'))
    best_metric['cat'] = cat
    io.write_json(best_metric, (fmt % 'best_metric.json'))
    if cfg.save_pth:
        torch.save(homan, (fmt % 'model.pth'))
        torch.save(weights, (fmt % 'weights.pth'))
        torch.save([list(v) for v in results], (fmt % 'results.pth'))

    if cfg.action_video.save:
        frames = make_compare_video(
            homan, global_cam, global_images=images,
            render_frames=cfg.action_video.render_frames)
        action_cilp = editor.ImageSequenceClip(frames, fps=5)
        action_cilp.write_videofile(fmt % 'action.mp4')

    for criterion in ['iou', 'collision', 'min_dist']:
        sign = 1 if criterion == 'iou' else -1
        final_score = \
            torch.softmax(torch.as_tensor([sign * getattr(v, criterion) for v in results]), 0)
        idx = final_score.argmax()
        R, t, s = results[idx].R, results[idx].t, results[idx].s
        homan.set_obj_transform(
            translations_object=t,
            rotations_object=R,
            scale_object=s)
        fig = homan.render_grid(obj_idx=0, with_hand=True, low_reso=False)
        fig.savefig(fmt % f'output_{criterion}.png')
        plt.close(fig)


if __name__ == '__main__':
    main()
