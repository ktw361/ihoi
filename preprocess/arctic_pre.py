import os
import os.path as osp
from PIL import Image
import tqdm
import numpy as np
import trimesh
import cv2

from nnutils.handmocap import get_hand_faces

from libzhifan import epylab, io, geometry
from libzhifan.geometry import visualize_mesh, SimpleMesh
from libzhifan.geometry import CameraManager, projection


low_reso = (728, 520)
orig_reso = (2800, 2000)

def reduce_image(subject, seq, view_id):
    """ reduce image resolution """
    data_root = 'arctic_data/'
    src_dir = osp.join(data_root, 'images', subject, seq, str(view_id))
    dst_dir = osp.join(data_root, 'images_low', subject, seq, str(view_id))
    os.makedirs(dst_dir, exist_ok=True)
    for image in tqdm.tqdm(os.listdir(src_dir)):
        src = osp.join(src_dir, image)
        src = Image.open(src)
        dst = osp.join(dst_dir, image)
        src.resize(low_reso).save(dst)

def bbox_from_mask(mask):
    """ 2d bbox. ret: [x0, y0, w, h] """
    _is, _js = np.where(mask == 1)
    if _is.size == 0:
        return None
    x0 = _js.min()
    w = _js.max() - x0
    y0 = _is.min()
    h = _is.max() - y0
    return np.asarray([x0, y0, w, h])


def convert(subject, seq, view_id, obj_name):
    """
    mask: 0 bg, 1 object, 2 left hand, 3 right hand
    """
    assert view_id == 0, "must be ego camera"
    os.makedirs(f'arctic_outputs/masks_low/{subject}/{seq}/{view_id}', exist_ok=True)
    os.makedirs(f'arctic_outputs/bboxes_low/{subject}/{seq}/{view_id}', exist_ok=True)
    palette = Image.open('/media/skynet/DATA/Datasets/visor-dense/meta_infos/00000.png').getpalette()
    proc_data = np.load(f'arctic_outputs/processed_verts/seqs/{subject}/{seq}.npy', allow_pickle=True).item()
    misc = io.read_json('arctic_data/meta/misc.json')
    intris_mat = misc[subject]['intris_mat'][view_id]
    intris_mat = np.asarray(intris_mat)
    # world2cam = misc[subject]['world2cam'][view_id]
    # world2cam = np.asarray(world2cam)
    # [ego, 8 views, distort ego]
    vl = proc_data['cam_coord']['verts.left']  # n_frames, views, 778, 3
    vr = proc_data['cam_coord']['verts.right']
    fl = get_hand_faces('left').cpu().numpy()
    fr = get_hand_faces('right').cpu().numpy()
    vo = proc_data['cam_coord']['verts.object']

    num_frame = vl.shape[0]
    bboxes = dict(left=[], right=[], object=[])
    os.makedirs(f'arctic_outputs/masks_low/{subject}/{seq}/{view_id}', exist_ok=True)
    for frame in tqdm.tqdm(range(num_frame)):
        left = SimpleMesh(verts=vl[frame, view_id], faces=fl, tex_color=(0, 1.0, 0))
        right = SimpleMesh(verts=vr[frame, view_id], faces=fr, tex_color=(0, 0, 1.0))
        mesh = trimesh.load(f'arctic_data/meta/object_vtemplates/{obj_name}/mesh.obj')
        obj = SimpleMesh(verts=vo[frame, view_id], faces=mesh.faces, tex_color=(1.0, 0, 0))

        # mesh_data = [left, obj, right] # visualize_mesh(mesh_data, show_axis=True, viewpoint='nr').show()
        K_ego = proc_data['params']['K_ego'][frame]
        w_ratio = low_reso[0] / orig_reso[0]
        h_ratio = low_reso[1] / orig_reso[1]
        fx = K_ego[0, 0] * w_ratio
        fy = K_ego[1, 1] * h_ratio
        cx = K_ego[0, 2] * w_ratio
        cy = K_ego[1, 2] * h_ratio
        cam_manager = CameraManager(fx, fy, cx, cy, img_h=low_reso[1], img_w=low_reso[0])

        proj_method = dict(name='pytorch3d', coor_sys='nr', in_ndc=False)
        rend = projection.perspective_projection_by_camera(
            [left, right, obj],
            camera=cam_manager,
            method=proj_method,
            image=np.zeros([low_reso[1], low_reso[0], 3], dtype=np.uint8))
        sil_o = rend[:, :, 0] > 0.25
        sil_l = rend[:, :, 1] > 0.25
        sil_r = rend[:, :, 2] > 0.25
        canvas = np.zeros((low_reso[1], low_reso[0]), dtype=np.uint8)
        canvas[sil_o > 0] = 1
        canvas[sil_l > 0] = 2
        canvas[sil_r > 0] = 3
        bbox_l = bbox_from_mask(sil_l)
        bbox_r = bbox_from_mask(sil_r)
        bbox_o = bbox_from_mask(sil_o)
        mask_path = f'arctic_outputs/masks_low/{subject}/{seq}/{view_id}/{frame:05d}.png'
        # cv2.imwrite(mask_path, canvas)
        img = Image.fromarray(canvas)
        img.putpalette(palette)
        img.save(mask_path)
        bboxes['left'].append(bbox_l)
        bboxes['right'].append(bbox_r)
        bboxes['object'].append(bbox_o)
    np.save(f'arctic_outputs/bboxes_low/{subject}/{seq}/{view_id}.npy', bboxes, allow_pickle=True)


if __name__ == '__main__':
    subject = 's01'
    view_id = 0
    seq = 'ketchup_grab_01'
    convert(subject, seq, view_id, 'ketchup')
