import os
import torch
import numpy as np
import trimesh
from typing import NamedTuple

from libzhifan.geometry import SimpleMesh


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
                 return_simplemesh=False):
        self.obj_models_root = obj_models_root
        self.return_simplemesh = return_simplemesh
        self.obj_models_cache = dict()

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
            obj_path = os.path.join(self.obj_models_root, f"{name}.obj")
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
            