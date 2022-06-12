import numpy as np 
from PIL import Image
import torch
import trimesh

from obj_pose.pose_optimizer import find_optimal_pose
import neural_renderer as nr


def main():
    image = './test_data/30150.png'
    image = np.asarray(Image.open(image), dtype=np.uint8)
    mask = './test_data/30150_mask.npy'
    mask = np.load(mask)
    # mask = np.asarray(Image.open(mask), dtype=np.float32)
    bbox = np.array([154.99997, 201.22983, 226.77412, 237.54033], dtype=np.float32)
    square_bbox = np.float32([113.985825, 165.59879 , 308.80243 , 308.80243 ])
    image_size = (640, 640, 3)
    K = np.float32([[490.32263,    0., -126.09146],
                    [0.,  490.32263,  522.9073],
                    [0.,    0.,    1.]])

    sort_best = False
    num_initalizations = 400

    obj_path = './weights/obj_models/plate.obj'
    scale = 0.3
    obj = trimesh.load(obj_path, force='mesh')
    verts = np.float32(obj.vertices)
    verts = verts - verts.mean(0)
    # inscribe in 1-radius sphere
    verts = verts / np.linalg.norm(verts, 2, 1).max() * scale / 2

    verts = torch.from_numpy(verts).cuda()
    faces = torch.from_numpy(obj.faces).cuda()

    model = find_optimal_pose(
        vertices=verts,
        faces=faces,
        mask=mask,
        bbox=bbox,
        square_bbox=square_bbox,
        image_size=image_size,
        K=K,
        num_initializations=num_initalizations,
        image=image,

        debug=True,
        sort_best=sort_best,
        viz=True,
    )


def nr_test():
    Rot = torch.as_tensor([
        [[ 0.5976,  0.4560,  0.6594],
         [-0.7621,  0.5786,  0.2906],
         [-0.2490, -0.6762,  0.6933]]], device='cuda:0')
    tra = torch.as_tensor([[ 0.4354, -0.2153,  0.5414]])
    obj_path = './weights/obj_models/plate.obj'
    scale = 0.3
    obj = trimesh.load(obj_path, force='mesh')
    verts = np.float32(obj.vertices)
    verts = verts - verts.mean(0)
    # inscribe in 1-radius sphere
    verts = verts / np.linalg.norm(verts, 2, 1).max() * scale / 2

    verts = torch.from_numpy(verts).cuda()
    faces = torch.from_numpy(obj.faces).cuda()

    device = 'cuda'
    K = torch.as_tensor([
        [[ 1.5878,  0.0000, -0.7794],                                                                                                                                                                                                         
         [ 0.0000,  1.5878,  1.1551],                                                                                                                                                                                                         
         [ 0.0000,  0.0000,  1.0000]]], device='cuda:0') 
    nr.renderer.Renderer(
        image_size=256,
        K=K,
        rot = torch.eye(3).unsqueeze(0).to(device),
        trans = torch.zeros(1, 3).to(device),
        orig_size=1,
        anti_aliasing=False,
    )


if __name__ == '__main__':
    main()