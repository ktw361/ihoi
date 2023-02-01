import os
from hydra.utils import to_absolute_path
import torch
import numpy as np
import trimesh
from typing import NamedTuple

from libzhifan.geometry import SimpleMesh
from libzhifan import io


class MeshHolder(NamedTuple):
    vertices: torch.Tensor
    faces: torch.Tensor


class OBJLoader:

    OBJ_SCALES = dict(
        plate=0.3,
        cup=0.12,
        can=0.2,
        mug=0.12,
        bowl=0.12,
        bottle=0.2
    )


    def __init__(self,
                 obj_models_root='./weights/obj_models',
                 load_simplified=True):
        self.obj_models_root = to_absolute_path(obj_models_root)
        self.obj_parts_root = to_absolute_path('./weights')
        self.obj_models_cache = dict()
        self.load_simplified = load_simplified

    def load_obj_by_name(self, name, return_mesh=False):
        """
        Args:
            return_mesh:
                if True, return a SimpleMesh object

        Returns:
            a Mesh Object that has attributes:
                `vertices` and `faces`
        """
        if name not in self.obj_models_cache:
            if self.load_simplified:
                obj_name = f"{name}_simplified.obj"
            else:
                obj_name = f"{name}.obj"
            obj_path = os.path.join(self.obj_models_root, obj_name)
            obj_scale = self.OBJ_SCALES[name]
            obj = trimesh.load(obj_path, force='mesh')
            verts = np.float32(obj.vertices)
            verts = verts - verts.mean(0)
            verts = verts / np.linalg.norm(verts, 2, 1).max() * obj_scale / 2
            obj_mesh = MeshHolder(vertices=verts, faces=obj.faces)
            self.obj_models_cache[name] = obj_mesh

        obj_mesh = self.obj_models_cache[name]

        if return_mesh:
            obj_mesh = SimpleMesh(obj_mesh.vertices, obj_mesh.faces)

        return obj_mesh

    def load_part_by_name(self, name):
        part = io.read_json(os.path.join(self.obj_parts_root, f"{name}_regions.json"))
        return part['verts'], part['faces']
