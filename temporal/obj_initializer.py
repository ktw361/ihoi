from typing import NamedTuple
from functools import reduce
import torch
from pytorch3d.transforms import matrix_to_rotation_6d

from obj_pose.obj_loader import OBJLoader
from homan.math import avg_matrix_approx
from homan.contact_prior import get_contact_regions
from homan.utils.geometry import (
    compute_random_rotations, generate_rotations_o2h
)
from datasets.epic_clip_v3 import DataElement
from homan.mvho_forwarder import LiteHandModule
from temporal.utils import (
    estimate_obj_scale, estimate_obj_depth,
    estimate_obj_xy
)


_obj_loader = OBJLoader()
_contact_regions = get_contact_regions()

class InitializerInput(NamedTuple):
    train_size: int  # T
    global_bbox: torch.Tensor  # (N*T, 4)
    global_cam: torch.Tensor   # (N*T, 3, 3)
    local_cam_mat: torch.Tensor
    base_rotation: torch.Tensor  # (N*T)
    base_translation: torch.Tensor  # or (N*T)
    v_hand_global: torch.Tensor  # (N*T)  camera space
    v_hand_local: torch.Tensor  # (N*T) In hand space
    obj_vertices: torch.Tensor  # (V, 3)
    obj_faces: torch.Tensor  # (F, 3)

    @staticmethod
    def prepare_input(hand_data: LiteHandModule.HandData, 
                      raw_input: DataElement,
                      train_size: int,
                      src_inds: torch.Tensor,):
        """
        Args:
            hand_data: hold (N*T)
        """
        global_bbox = raw_input.obj_bboxes
        global_cam = raw_input.global_camera.get_K()
        global_bbox = InitializerInput.gather0d(global_bbox, src_inds)
        global_cam = InitializerInput.gather0d(global_cam, src_inds)
        obj_mesh = _obj_loader.load_obj_by_name(raw_input.cat, return_mesh=False)
        obj_vertices = torch.as_tensor(obj_mesh.vertices, device='cuda')
        obj_faces = torch.as_tensor(obj_mesh.faces, device='cuda')

        init_input = InitializerInput(
            train_size=train_size, global_bbox=global_bbox, 
            global_cam=global_cam,
            local_cam_mat=hand_data.camintr, base_rotation=hand_data.rot_mat_hand,
            base_translation=hand_data.translations_hand,
            v_hand_global=hand_data.v_hand_global,
            v_hand_local=hand_data.v_hand_local,
            obj_vertices=obj_vertices, obj_faces=obj_faces)
        
        return init_input

    @staticmethod
    def gather0d(source, indices: torch.Tensor):
        """ 
        Args:
            source: (L, F1, F2, ...)
            indices: (N*T)
        Returns:
            (N*T, F1, F2, ...)
        """
        feature_shape = source.shape[1:]
        dummy_shape = (1,) * len(feature_shape)
        N, T = indices.shape
        indices = indices.view(-1, *dummy_shape).expand(-1, *feature_shape)
        out = torch.gather(source, dim=0, index=indices)
        return out


class ObjectPoseInitializer:
    """ Multi-frame(view) object pose initializer 
    Initialization order:
    rotation -> scale -> translation

    First generate num_init(N) rotations, this can depend on base_rotations.
    For each N, scale and transl depends on this rotation.
    """

    def __init__(self,
                 rot_init: dict,
                 scale_init_method: str,
                 transl_init_method: str,
                 device='cuda'):
        """
        Args:
            verts_obj: [N, 3]
        """
        self.rot_init = rot_init
        self.scale_init_method = scale_init_method
        self.transl_init_method = transl_init_method
        self.device = device

        self.num_inits = ObjectPoseInitializer.read_num_inits(rot_init)

        """ inputs """
        self._in_data = None
        """ internal """
        self._R_o2h = None  # (N, 3, 3)
        self._V_rotated = None  # (N, T, V, 3)
        """ outputs """
        self._R_o2h_6d = None  # (N, 6)
        self._translation_inits = None  # (N, 3)
        self._scale_inits = None  # (N, 1) or (N, 3)
    
    @staticmethod
    def read_num_inits(rot_init: dict):
        if rot_init['method'] == 'upright' or rot_init['method'] == 'spiral':
            return rot_init['num_sphere_pts'] * rot_init['num_sym_rots']
        if rot_init['method'] == 'random':
            return rot_init['num_inits']
    
    def set_input(self, in_data: InitializerInput):
        self._in_data = in_data

    def init_rot(self):
        """ Initialize N rotations from N*T observations

        Returns: 
        in base space, col-vec
            rotation6d: (N, 6)
            V_rotated: (N, T, V, 3)
        """
        rot_init = self.rot_init
        base_rotations = self._in_data.base_rotation
        verts_obj = self._in_data.obj_vertices
        T = self._in_data.train_size
        N = self.num_inits
        if rot_init['method'] == 'upright':
            avg_base_rotation = [None for _ in range(N)]
            for n in range(N):
                avg_base_rotation[n] = avg_matrix_approx(
                    base_rotations[n*T:(n+1)*T, ...]).view(1, 3, 3)
            avg_base_rotation = torch.cat(avg_base_rotation, dim=0)
            R_o2h, _ = generate_rotations_o2h(
                rot_init, base_rotations=avg_base_rotation,
                device=self.device)
        else:
            R_o2h, _ = generate_rotations_o2h(
                rot_init, base_rotations=None, device=self.device)

        """ V_out = (V_model @ R + T) @ R_base + T_base """
        R_o2h_row = R_o2h.permute(0, 2, 1)  # (N, 3, 3)
        base_rot_row = base_rotations.permute(0, 2, 1).view(N, T, 3, 3)
        V_rotated = torch.matmul(
            (verts_obj @ R_o2h_row).view(N, 1, -1, 3),
            base_rot_row)  # (N, T, V, 3)

        self._R_o2h = R_o2h
        self._V_rotated = V_rotated
        self._R_o2h_6d = matrix_to_rotation_6d(R_o2h)
        return R_o2h, V_rotated
    
    def init_scale(self):
        scale_init_method = self.scale_init_method
        num_inits = self.num_inits
        if scale_init_method == 'one':
            scale_inits = torch.ones([num_inits], device=self.device)
        elif scale_init_method == 'xyz':
            scale_inits = torch.new_ones([num_inits, 3], device=self.device)
        elif scale_init_method == 'est':
            global_bboxes = self._in_data.global_bbox
            verts_hand_global = self._in_data.v_hand_global
            global_cam_mat = self._in_data.global_cam

            N = self.num_inits
            scale_inits = [None for _ in range(N)]
            T = self._in_data.train_size
            V_rotated = self._V_rotated
            V_rotated_tn = V_rotated.permute(1, 0, 2, 3)  # (N, T) => (T, N)
            for n in range(num_inits):
                scale_inits[n] = estimate_obj_scale(
                    global_bboxes[n*T:(n+1)*T, ...], 
                    verts_hand_global[n*T:(n+1)*T, ...], 
                    V_rotated_tn[:, [n], :, :],
                    global_cam_mat[n*T:(n+1)*T, ...])
            scale_inits = torch.cat(scale_inits, dim=0)  # (N, )
            V_rotated = V_rotated * scale_inits.view(N, 1, 1, 1).expand(N, T, 1, 1)
            self._V_rotated = V_rotated
        else:
            raise ValueError()
        
        self._scale_inits = scale_inits
    
    def init_transl(self):
        transl_init_method = self.transl_init_method
        v_hand_local = self._in_data.v_hand_local
        num_inits = self.num_inits
        # For now, use for loop
        if transl_init_method == 'zero':
            translation_inits = torch.zeros([num_inits, 1, 3], device=self.device)
        elif transl_init_method == 'fingers':
            num_priors = 5
            v_inds = reduce(lambda a, b: a + b, _contact_regions.verts[:num_priors], [])
            finger_center = v_hand_local[:, v_inds, :]  # (T, 3)
            translation_inits = finger_center.mean(dim=(0, 1)).view(1, 1, 3).tile(num_inits, 1, 1) 
        elif transl_init_method == 'mask':
            raise NotImplementedError
            global_bboxes = self._in_data.global_bbox
            global_cam_mat = self._in_data.global_cam
            local_cam_mat = self._in_data.local_cam
            V_rotated = self._V_rotated
            V_rotated_tn = V_rotated.permute(1, 0, 2, 3)  # (N, T) => (T, N)
            z_o2c = estimate_obj_depth(global_bboxes, V_rotated, local_cam_mat)  # (T, N_init), obj-to-cam
            x_o2c, y_o2c = estimate_obj_xy(global_bboxes, V_rotated, global_cam_mat, z_o2c)  # (T, N_init), obj-to-cam
            est_translations = torch.cat([x_o2c, y_o2c, z_o2c], dim=-1).view(bsize, num_init, 1, 3)
            translations_init = torch.einsum(
                'bnij,bkj->bnik', 
                est_translations - base_translation.unsqueeze(1), base_rotation)
            translations_init = translations_init.mean(dim=0)
        else:
            raise ValueError()
        
        self._translation_inits = translation_inits

    def init_pose(self, in_data: InitializerInput):
        """ Initialize N poses from N*T observations

        Returns: 
        in base space, col-vec
            rotation6d: (N, 6)
            translation: (N, 3)
            scale: (N, 1)
        """
        self.set_input(in_data)
        self.init_rot()
        self.init_scale()
        self.init_transl()
        return self._R_o2h_6d, self._translation_inits, self._scale_inits
