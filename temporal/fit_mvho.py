import os
import hydra
from omegaconf import DictConfig, OmegaConf
import tqdm
import numpy as np
import torch
import logging
from moviepy import editor
import matplotlib.pyplot as plt

from nnutils.handmocap import extract_forwarder_input
from datasets.epic_clip_v3 import EpicClipDatasetV3
from homan.mvho_forwarder import MVHOVis, LiteHandModule
from temporal.optim_plan import (
    optimize_hand, smooth_hand_pose
)
from temporal.obj_initializer import ObjectPoseInitializer, InitializerInput
from temporal.optim_multiview import EvalHelper, multiview_optimize
from temporal.post_refinement import load_homan_from_mvho, optimize_post
from temporal.visualize import make_compare_video

from libzhifan import io


@hydra.main(config_path='../config', config_name='conf_multiview')
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    log = logging.getLogger(__name__)

    assert cfg.dataset.version == 'v3'
    sample_frames = cfg.optim_mv.num_source
    dataset = EpicClipDatasetV3(
        image_sets=cfg.dataset.image_sets,
        sample_frames=sample_frames,
        show_loading_time=True)
    eval_dataset = EpicClipDatasetV3(
        image_sets=cfg.dataset.image_sets,
        sample_frames=cfg.optim_mv.num_eval,
        show_loading_time=True)

    if cfg.debug_locate is not None:
        index = dataset.locate_index_from_output(cfg.debug_locate)
        fit_scene(dataset, eval_dataset, index, cfg=cfg)
        return
    if cfg.debug_index is not None:
        fit_scene(dataset, eval_dataset, cfg.debug_index, cfg=cfg)
        return

    for index in tqdm.trange(cfg.index_from, min(cfg.index_to, len(dataset))):
        try:
            r = fit_scene(dataset, eval_dataset, index, cfg=cfg)
            if r is not None:
                log.info(f"{r} at index [{index}]: {dataset.data_infos[index]}")
            else:
                log.info(f"Succeed at index [{index}]: {dataset.data_infos[index]}")
        except Exception as e:
            log.info(f"Failed at index [{index}]: {dataset.data_infos[index]}. Reason: {e}")
            continue


def fit_scene(dataset,
              eval_dataset,
              index: int,
              cfg: DictConfig = None):
    """
    Pipeline:
        L: source size, 100
        E: eval size, 30
        T: train size, 3
        N: number of initializations, 400
        Np: number of initailizations in parallel, 40

    Input: L images, E eval images
    L hand data <- frankmocap()
    src_inds: (N*T) , sample T for each N
    hand_data: (N*T) <- sample hand_data with src_inds
    obj_data:  (N*T) <- sample obj_data with src_inds
    init_poses: (N*T)

    homan = MVHOVis()
    for p in range(N // Np):
        hand_data_batch: (Np*T) <- hand_data[p*Np*T:(p+1)*Np*T]
        obj_data_batch:  (Np*T) <- obj_data[p*Np*T:(p+1)*Np*T]
        init_poses_batch: (Np*T) <- init_poses[p*Np*T:(p+1)*Np*T]
        homan.set_input(hand_data, obj_data)
        homan.set_init_poses(init_poses_batch)
        optimize(homan.obj_poses)
        homan.set_input(hand_eval, obj_eval)
        best_pose = homan.best_eval_pose()

    Args:
        cfg: see config/conf.yaml
    """
    np.random.seed(0)
    torch.random.manual_seed(0)

    device = 'cuda'
    info = dataset.data_infos[index]
    if cfg.only_cat is not None and info.cat != cfg.only_cat:
        return 'skip cat'
    fmt = f'{info.vid}_{info.start}_{info.end}_%s'
    if cfg.skip_existing and os.path.exists(fmt % 'model.pth'):
        return 'skip'
    input_data = dataset[index]
    images, hand_bbox_dicts, side, obj_bboxes, \
        hand_masks, obj_masks, cat, global_cam = input_data
    if len(images) < 10:
        print(f'Skip {info.vid} due to too few frames ({len(images)})')
        return 'skip too few frames'

    # ihoi_cam_mat and image_path set in prepare_input()
    ihoi_cam_nr_mat, ihoi_cam_mat, image_patch, \
        hand_rotation_6d, hand_translation, \
        mano_pca_pose, pred_hand_betas, hand_mask_patch, obj_mask_patch = \
        extract_forwarder_input(
            input_data, ihoi_box_expand=cfg.preprocess.ihoi_box_expand)

    """ Frankmocap on all source frames """
    lite_hand = LiteHandModule()
    lite_hand_params = LiteHandModule.LiteHandParams(
        ihoi_cam_nr_mat, hand_rotation_6d, hand_translation,
        side, mano_pca_pose, pred_hand_betas, hand_mask_patch)
    lite_hand.set_hand_params(lite_hand_params)
    # Step 1: Interpolate pca_pose Step 2: Optimize hand_mask
    if cfg.homan.optimize_hand:
        print('Smooth hand pose and optimize hand')
        lite_hand = smooth_hand_pose(lite_hand, lr=0.1)
        lite_hand = optimize_hand(lite_hand, verbose=False)

    optim_cfg = cfg.optim_mv
    """ Get training indices"""
    num_source = len(ihoi_cam_nr_mat)

    num_inits_parallel = optim_cfg.num_inits_parallel

    if cat == 'cup' or cat == 'mug':
        num_inits_parallel = num_inits_parallel // 2
    rot_init = cfg.homan.rot_init[cat]
    if rot_init['method'] == 'spiral' or rot_init['method'] == 'upright':
        rot_init['num_sphere_pts'] = optim_cfg.num_inits // rot_init['num_sym_rots']
    scale_init = cfg.homan.scale_init[cat]
    num_inits = ObjectPoseInitializer.read_num_inits(rot_init)
    train_size = min(optim_cfg.train_size, num_source)
    src_inds = torch.ones([num_inits, num_source]).multinomial(
        train_size, replacement=False)

    """ Get All source hand data and obj target """
    hand_data = lite_hand[ src_inds ]
    target_masks_object = lite_hand.gather0d(obj_mask_patch, src_inds)

    """ init_pose() """
    init_input = InitializerInput.prepare_input(
        hand_data, input_data, train_size=train_size, src_inds=src_inds)
    obj_initializer = ObjectPoseInitializer(
        rot_init, scale_init,
        cfg.homan.transl_init_method)
    R_o2h_6d, translation_inits, scale_inits = obj_initializer.init_pose(init_input)

    eval_helper = EvalHelper()
    eval_helper.set_eval_data(eval_dataset, index, cfg, side,
                              optimize_eval_hand=cfg.homan.optimize_eval_hand)

    """ optimize(homan.obj_poses) """
    mvho = MVHOVis()  # homan.set_ihoi_img_patch(image_patch)
    mvho.register_obj_buffer(
        verts_object_og=init_input.obj_vertices,
        faces_object=init_input.obj_faces,
        scale_mode=cfg.homan.scale_mode)
    for e in tqdm.trange(num_inits // num_inits_parallel):
        nt_start = e * num_inits_parallel * train_size
        nt_end = (e+1) * num_inits_parallel * train_size
        mvho.set_size(num_inits_parallel, train_size)  # eval will set this to sth. else
        mvho.set_hand_data(hand_data[nt_start:nt_end, ...])
        n_start = e * num_inits_parallel
        n_end = (e+1) * num_inits_parallel
        mvho.set_obj_transform(
            translations_object=translation_inits[n_start:n_end, ...],
            rotations_object=R_o2h_6d[n_start:n_end, ...],
            scale_object=scale_inits[n_start:n_end, ...])
        mvho.set_obj_target(
            target_masks_object[nt_start:nt_end, ...], check_shape=False)

        mvho = multiview_optimize(mvho, optim_cfg)
        eval_helper.register_batch(mvho, e, num_inits_parallel)

    mvho, best_metric = eval_helper.decide_best_homan(
        mvho, optim_cfg.criterion)

    """ post refinement """
    if cfg.post_refine:
        print('Start Post refinement')
        homan = load_homan_from_mvho(
            eval_helper.eval_input, mvho, cfg,
            mano_pca_pose=eval_helper.eval_mano_pca_pose,
            mano_betas=eval_helper.eval_mano_betas)
        homan.register_combined_target()
        pre_metrics = homan.eval_metrics(unsafe=True, avg=True)
        homan = optimize_post(homan, steps=200)
        post_metrics = homan.eval_metrics(unsafe=True, avg=True)
        torch.save(homan, (fmt % 'post.pth'))
        io.write_json(pre_metrics, (fmt % 'pre.json'))
        io.write_json(post_metrics, (fmt % 'post.json'))
        print("Post refinement done")
        if cfg.action_video.save:
            frames = make_compare_video(
                homan, eval_helper.eval_input.global_camera, 
                global_images=eval_helper.eval_input.images,
                render_frames='all')
            action_cilp = editor.ImageSequenceClip(frames, fps=5)
            action_cilp.write_videofile(fmt % 'post.mp4')

    """ saving """
    mvho.render_grid(pose_idx=0, with_hand=False,
                      low_reso=False, overlay_gt=True).savefig(fmt % 'eval.png')
    plt.clf()
    # mvho.to_scene(pose_idx=0, show_axis=False).export((fmt % 'mesh.obj'))
    io.write_json(best_metric, (fmt % 'best_metric.json'))
    if cfg.save_pth:
        torch.save(mvho, (fmt % 'model.pth'))
        # torch.save([list(v) for v in eval_helper.eval_results], (fmt % 'results.pth'))

    if cfg.action_video.save:
        frames = eval_helper.make_compare_video(mvho)
        action_cilp = editor.ImageSequenceClip(frames, fps=5)
        action_cilp.write_videofile(fmt % 'pre.mp4')

    for criterion in ['iou', 'max_min_dist']:
        mvho, best_metric = eval_helper.decide_best_homan(
            mvho, criterion)
        fig = mvho.render_grid(pose_idx=0, with_hand=True, low_reso=False)
        fig.savefig(fmt % f'output_{criterion}.png')
        plt.close(fig)


if __name__ == '__main__':
    main()
