import os
import os.path as osp
import torch
import cv2
import numpy as np
from libzhifan import epylab
from datasets.epic_clip_v3 import EpicClipDatasetV3

from libzhifan.geometry import SimpleMesh, visualize_mesh, projection, CameraManager
from libzhifan.geometry import visualize as geo_vis


def save_wrapper(img, name):
    epylab.eimshow(img, title='')
    epylab.axis('off')
    epylab.savefig(
        f'{name}.png', bbox_inches='tight', dpi=300)

def highlight_with_mask(img: np.ndarray, 
                        mask: np.ndarray, 
                        weight=0.5,
                        mul255=True):
    """
    Args:
        img: (H, W, 3)
        mask: (H, W) 1 for fg and 0 for bg
    Returns:
        img
    """
    canvas = np.ones_like(img)
    if mul255:
        canvas = canvas * 255
    canvas[mask==1] = img[mask==1]
    ret = cv2.addWeighted(canvas, weight, img, 1-weight, 0)
    return ret
    # idx = 29
    # img = data_elem.images[idx]
    # m = np.logical_or(
    #     data_elem.hand_masks[idx].numpy() > 0,
    #     data_elem.object_masks[idx].numpy() > 0)
    # hl = highlight_with_mask(img, m, 0.6)
    # save_wrapper(hl, 'P02_03_32033_32113_ed')

def render_dualview(homan, scene_idx, obj_idx=0, **mesh_kwargs) -> np.ndarray:
    """
    Returns:
        (H, W, 3)
    """
    rend_size = 256
    mhand, mobj = homan.get_meshes(scene_idx=scene_idx, obj_idx=obj_idx, **mesh_kwargs)
    # front = projection.project_standardized(
    #     [mhand, mobj],
    #     direction='+z',
    #     image_size=rend_size,
    #     method=dict(
    #         name='pytorch3d',
    #         coor_sys='nr',
    #         in_ndc=False
    #     )
    # )
    left = projection.project_standardized(
        [mhand, mobj],
        direction='+x',
        image_size=rend_size,
        method=dict(
            name='pytorch3d',
            coor_sys='nr',
            in_ndc=False
        )
    )
    back = projection.project_standardized(
        [mhand, mobj],
        direction='-z',
        image_size=rend_size,
        method=dict(
            name='pytorch3d',
            coor_sys='nr',
            in_ndc=False
        )
    )
    # return np.hstack([left, back])
    return np.vstack([left, back])


class Shower:
    plate = [
        [
        'P06_13_14212_14244_post.pth', 0, 29,
        ]
    ]
    def __init__(self):
        image_sets = '/home/skynet/Zhifan/epic_analysis/hos/tools/model-input-Feb03.json'
        sample_frames = 30
        self.dataset = EpicClipDatasetV3(
            image_sets=image_sets, sample_frames=sample_frames,
            show_loading_time=True
        )
    
    def find_model(self, vid):
        sources = [
            osp.join('outputs', v)
            for v in 
            ['2023-03-05/Rest/', '2023-03-06/Rest-400-600', '2023-03-06/Rest-600-813',]
        ]
        vid = '_'.join(vid.split('_')[:4])
        for src in sources:
            for f in os.listdir(src):
                if vid in f and 'post.pth' in f:
                    return osp.join(src, f)
        return None
    
    def get_model(self, path):
        base = osp.basename(path)
        index = self.dataset.locate_index_from_output(base)
        e = self.dataset[index]
        homan = torch.load(self.find_model(path))
        return e, homan

    def show_model(self, e, homan, st=0, ed=29, mid=14,
                   vertical=False, with_dual=True,
                   masking: float = None):
        """
        a1 | b1 | triview
        a2 | b2 | triview
        """
        obj_idx = 0
        images = [v for v in e.images]

        a1 = e.images[st] / 255
        a2 = e.images[ed] / 255
        b1 = homan.render_global(
            e.global_camera, images,
                scene_idx=st, obj_idx=obj_idx)
        b2 = homan.render_global(
            e.global_camera, images,
                scene_idx=ed, obj_idx=obj_idx)
        if masking is not None:
            m1 = np.logical_or(
                e.hand_masks[st].numpy() > 0,
                e.object_masks[st].numpy() > 0)
            m2 = np.logical_or(
                e.hand_masks[ed].numpy() > 0,
                e.object_masks[ed].numpy() > 0)
            a1 = highlight_with_mask(a1, m1, masking, mul255=False)
            a2 = highlight_with_mask(a2, m2, masking, mul255=False)
            images[st] = highlight_with_mask(images[st], m1, masking, mul255=True)
            images[ed] = highlight_with_mask(images[ed], m2, masking, mul255=True)
            b2 = images[ed]
            b1 = homan.render_global(
                e.global_camera, images,
                    scene_idx=st, obj_idx=obj_idx)
            b2 = homan.render_global(
                e.global_camera, images,
                    scene_idx=ed, obj_idx=obj_idx)

        if vertical:
            a = np.vstack([a1, a2])
            b = np.vstack([b1, b2])
            ab = np.hstack([a, b])
        else:
            a = np.hstack([a1, a2])
            b = np.hstack([b1, b2])
            ab = np.vstack([a, b])

        if with_dual:
            c = render_dualview(
                homan,scene_idx=mid, obj_idx=obj_idx)
            h, w = ab.shape[:2]
            ratio = h / c.shape[0]
            ww = int(c.shape[1] * ratio)
            hh = h
            c = cv2.resize(c, (ww, hh))
            out = np.hstack([ab, c])
        else:
            out = ab

        return out
    
    def save_view(self, e, homan, st=0, ed=29, mid=14,
                  vertical=False, with_dual=True,
                  masking: float = None,
                  name=None):
        out = self.show_model(
            e, homan, st, ed, mid, vertical,
            masking=masking, with_dual=with_dual)
        assert name is not None
        name = e.cat + '_' + name
        save_wrapper(out, name)