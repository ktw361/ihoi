from typing import NamedTuple
from collections import namedtuple
import tqdm
import torch

from homan.mvho_forwarder import MVHOVis, LiteHandModule
from nnutils.handmocap import extract_forwarder_input

ElementType = namedtuple(
    "ElementType", "iou collision min_dist R t s sample_indices")


class EvalHelper:
    """ This class
    - perform evaluation, (helps summing N*T into N, possibly with for-loop due to memory)
    - stores best results
    - helps to call visualize figures and metrics
    """
    def __init__(self):
        self.num_eval = None
        # Don't do rolling selection, just store all results for now
        # self.R_best = None  # (1, 6)
        # self.T_best = None  # (1, 1, 3)
        # self.s_best = None  # (1, 1)
        # self.score_best = torch.as_tensor(np.inf)  # (,)
        # self.best_index = None  # int
        self.eval_results = []

        self.eval_hand_data = None
        self.eval_image_patch = None
        self.eval_target_masks_object = None

    @staticmethod
    def get_eval_data(eval_dataset, index, cfg, side):
        """ Get eval data 
        Args:
            cfg: full config
        """
        eval_input = eval_dataset[index]
        # images, hand_bbox_dicts, side, obj_bboxes, hand_masks, obj_masks, cat, global_cam = input_data

        eval_ihoi_cam_nr_mat, eval_ihoi_cam_mat, eval_image_patch, \
        eval_hand_rotation_6d, eval_hand_translation, \
        eval_mano_pca_pose, eval_pred_hand_betas, eval_hand_mask_patch, eval_obj_mask_patch = \
            extract_forwarder_input(
                eval_input, ihoi_box_expand=cfg.preprocess.ihoi_box_expand)
        num_eval = min(cfg.optim_multiview.num_eval, len(eval_ihoi_cam_nr_mat))

        # Frankmocap on all eval frames
        eval_hand = LiteHandModule()
        eval_hand_params = LiteHandModule.LiteHandParams(
            eval_ihoi_cam_nr_mat, eval_hand_rotation_6d, eval_hand_translation,
            side, eval_mano_pca_pose, eval_pred_hand_betas, eval_hand_mask_patch)
        eval_hand.set_hand_params(eval_hand_params)

        eval_inds = torch.arange(num_eval).view(-1, 1)
        eval_hand_data = eval_hand[ eval_inds ]
        eval_target_masks_object = eval_hand.gather0d(eval_obj_mask_patch, eval_inds)
        return eval_hand_data, eval_image_patch, eval_target_masks_object

    def set_eval_data(self, eval_dataset, index, cfg, side):
        """
        Args:
            eval_dataset: EpicClipV3
            index: same as train
            cfg: full cfg
            side: str
        """
        self.eval_hand_data, self.eval_image_patch, self.eval_target_masks_object = \
            EvalHelper.get_eval_data(eval_dataset, index, cfg, side)
        self.num_eval = min(cfg.optim_multiview.num_eval, len(self.eval_image_patch))

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

            iou = metrics['iou']                # bigger better
            collision = metrics['collision']    # smaller better
            min_dist = metrics['min_dist']      # smaller better
            mean_iou = iou.mean(0)
            mean_collision = collision.mean(0)
            mean_min_dist = min_dist.mean(0)
            element = ElementType(
                mean_iou.item(), mean_collision.item(), mean_min_dist.item(),
                R_train[[i]], T_train[[i]], s_train[[i]], None)
            self.eval_results.append(element)

    def decide_best_homan(self,
                          homan: MVHOVis,
                          criterion: str):
        results = self.eval_results
        sign = 1 if criterion == 'iou' else -1
        final_score = \
            torch.softmax(torch.as_tensor([sign * getattr(v, criterion) for v in results]), 0)

        best_idx = final_score.argmax()
        best_metric = {
            'iou': results[best_idx].iou, 'collision': results[best_idx].collision,
            'min_dist': results[best_idx].min_dist}
        R, t, s = results[best_idx].R, results[best_idx].t, results[best_idx].s
        homan.set_obj_transform(
            translations_object=t,
            rotations_object=R,
            scale_object=s)
        # homan._check_shape_object(1)
        return homan, best_metric


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
        cfg: cfg.optim_multiview in config/conf.yaml
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
            # TODO: manual set loss=0 no grad?

            if torch.isnan(tot_loss) or torch.isinf(tot_loss):
                break
            tot_loss.backward()
            optimizer.step()
            loop.set_description(f"tot loss: {tot_loss.item():.3g}")
            loop.update()

    return homan
