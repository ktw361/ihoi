from typing import NamedTuple
from collections import namedtuple
import tqdm
import torch

from temporal.optim_plan import (
    optimize_hand, smooth_hand_pose
)
from homan.mvho_forwarder import MVHOVis, LiteHandModule
from nnutils.handmocap import extract_forwarder_input

ElementType = namedtuple(
    "ElementType", "iou collision max_min_dist R t s sample_indices")


class EvalHelper:
    """ This class
    - perform evaluation, (helps summing N*T into N, possibly with for-loop due to memory)
    - stores best results
    - helps to call visualize figures and metrics
    """
    def __init__(self):
        self.num_eval = None
        self.eval_results = []

        self.eval_input = None
        self.eval_mano_pca_pose = None
        self.eval_mano_betas = None

        self.eval_hand_data = None
        self.eval_image_patch = None
        self.eval_target_masks_object = None

        self.movie_global_cam = None  # For make_compare_video
        self.movie_images = None

    def get_eval_data(self, eval_dataset, index, cfg, side,
                      optimize_eval_hand=True):
        """ Get eval data
        Args:
            cfg: full config
        """
        eval_input = eval_dataset[index]
        images, hand_bbox_dicts, side, obj_bboxes, hand_masks, obj_masks, cat, global_cam = eval_input
        self.eval_input = eval_input

        eval_ihoi_cam_nr_mat, eval_ihoi_cam_mat, eval_image_patch, \
        eval_hand_rotation_6d, eval_hand_translation, \
        eval_mano_pca_pose, eval_pred_hand_betas, eval_hand_mask_patch, eval_obj_mask_patch = \
            extract_forwarder_input(
                eval_input, ihoi_box_expand=cfg.preprocess.ihoi_box_expand)
        num_eval = min(cfg.optim_mv.num_eval, len(eval_ihoi_cam_nr_mat))

        # Frankmocap on all eval frames
        eval_hand = LiteHandModule()
        eval_hand_params = LiteHandModule.LiteHandParams(
            eval_ihoi_cam_nr_mat, eval_hand_rotation_6d, eval_hand_translation,
            side, eval_mano_pca_pose, eval_pred_hand_betas, eval_hand_mask_patch)
        eval_hand.set_hand_params(eval_hand_params)

        if optimize_eval_hand:
            print('Eval: Smooth hand pose and optimize hand')
            eval_hand = smooth_hand_pose(eval_hand, lr=0.1)
            eval_hand = optimize_hand(eval_hand, verbose=False)

        self.eval_mano_pca_pose = eval_hand.mano_pca_pose.detach().clone()
        self.eval_mano_betas = eval_hand.mano_betas.detach().clone()

        eval_inds = torch.arange(num_eval).view(-1, 1)
        eval_hand_data = eval_hand[ eval_inds ]
        eval_target_masks_object = eval_hand.gather0d(eval_obj_mask_patch, eval_inds)
        return eval_hand_data, eval_image_patch, eval_target_masks_object, \
            global_cam, images

    def set_eval_data(self, eval_dataset, index, cfg, side,
                      optimize_eval_hand=True):
        """
        Args:
            eval_dataset: EpicClipV3
            index: same as train
            cfg: full cfg
            side: str
        """
        self.eval_hand_data, self.eval_image_patch, self.eval_target_masks_object,\
            self.movie_global_cam, self.movie_images = \
            self.get_eval_data(
                eval_dataset, index, cfg, side, optimize_eval_hand)
        self.num_eval = min(cfg.optim_mv.num_eval, len(self.eval_image_patch))

    def register_batch(self,
                       homan: MVHOVis,
                       epoch: int,
                       num_inits_parallel: int):
        R_train = homan.rotations_object.detach().clone()
        T_train = homan.translations_object.detach().clone()
        s_train = homan.scale_object.detach().clone()
        homan.set_size(1, self.num_eval)
        homan.set_ihoi_img_patch(self.eval_image_patch)
        homan.set_hand_data(self.eval_hand_data)
        homan.set_obj_target(self.eval_target_masks_object, check_shape=False)
        for i in range(num_inits_parallel):
            homan.set_obj_transform(
                translations_object=T_train[[i]],
                rotations_object=R_train[[i]],
                scale_object=s_train[[i]])
            with torch.no_grad():
                metrics = homan.eval_metrics()

            iou = metrics['oious']                # bigger better
            mean_iou = iou.mean(0)
            # collision = metrics['collision']    # smaller better
            max_min_dist = metrics['max_min_dist'] # smaller better
            element = ElementType(
                mean_iou.item(), 0, max_min_dist,
                R_train[[i]], T_train[[i]], s_train[[i]], None)
            self.eval_results.append(element)

    def decide_best_homan(self,
                          homan: MVHOVis,
                          criterion: str):
        """
        Args:
            criterion: 'iou' or 'pd_h2o' or 'pd_o2h', 'max_min_dist'
        """
        results = self.eval_results
        sign = 1 if criterion == 'iou' else -1
        final_score = \
            torch.softmax(torch.as_tensor([sign * getattr(v, criterion) for v in results]), 0)

        best_idx = final_score.argmax()
        R, t, s = results[best_idx].R, results[best_idx].t, results[best_idx].s
        homan.set_obj_transform(
            translations_object=t,
            rotations_object=R,
            scale_object=s)
        # pd_h2o, pd_o2h = homan.penetration_depth()
        # pd_h2o = pd_h2o.max().item()
        # pd_o2h = pd_o2h.max().item()
        best_metric = {
            'iou': results[best_idx].iou,
            # 'pd_h2o': pd_h2o,
            # 'pd_o2h': pd_o2h,
            'max_min_dist': results[best_idx].max_min_dist}
        return homan, best_metric

    def make_compare_video(self, homan):
        frames = homan.make_compare_video(
            self.movie_global_cam, global_images=self.movie_images, pose_idx=0)
        return frames


def multiview_optimize(homan: MVHOVis,
                       optim_cfg) -> MVHOVis:
    """
    homan stores the whole source_bank of <image, mask>,
    for each init pose, it sample a small number of frames, optimize them,
    and eval the metrics on eval_frames.
    Pose with best metric is chosen as the final result.

    source_bank should be distributed uniformly in input frames, e.g. at most N=100 frames.
    eval_frames should be a subset of source_bank, e.g. at most N=20 frames.
    each pose sees a small fraction of source_bank, e.g. N=3 frames.

    Args:
        cfg: cfg.optim_mv in config/conf.yaml
    """
    # Read out from config
    lr = optim_cfg.lr
    num_iters = optim_cfg.num_iters
    vis_interval = optim_cfg.vis_interval

    params = [
        homan.rotations_object,  # (Np*T,)
        homan.translations_object,
        homan.scale_object,
    ]
    optimizer = torch.optim.Adam([{
        'params': params,
        'lr': lr
    }])
    with tqdm.tqdm(total=num_iters, disable=not optim_cfg.iter_tqdm) as loop:
        for step in range(num_iters):
            optimizer.zero_grad()

            print_metric = (vis_interval > 0 and step % vis_interval == 0)
            tot_loss = homan.train_loss(
                optim_cfg=optim_cfg, print_metric=print_metric)

            if torch.isnan(tot_loss) or torch.isinf(tot_loss):
                break
            tot_loss.backward()
            optimizer.step()
            loop.set_description(f"tot loss: {tot_loss.item():.3g}")
            loop.update()

    return homan
