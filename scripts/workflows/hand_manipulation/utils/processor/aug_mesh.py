import numpy as np
import trimesh
import torch
import isaaclab.utils.math as math_utils
import open3d as o3d
import pymeshlab

import os


class MeshAugmentation:

    def __init__(self,
                 source_mesh_dir,
                 target_mesh_dir,
                 pymeshlab_iter=2,
                 downsample_points=None):
        self.source_mesh_dir = source_mesh_dir
        self.target_mesh_dir = target_mesh_dir
        self.pymeshlab_iter = pymeshlab_iter
        os.makedirs(self.target_mesh_dir, exist_ok=True)
        self.downsample_points = downsample_points

    def apply(self):
        for filename in os.listdir(self.source_mesh_dir):
            if filename.endswith(".obj"):
                source_path = os.path.join(self.source_mesh_dir, filename)
                target_path = os.path.join(self.target_mesh_dir, filename)

                # Load the mesh
                mesh = trimesh.load(source_path)

                # Apply transformations
                vertices, faces = self.augment_mesh(mesh)

                transformed_mesh = trimesh.Trimesh(
                    vertices=vertices[0].cpu().numpy(), faces=faces)

                # Save the augmented mesh
                print(target_path, transformed_mesh.vertices.shape)

                transformed_mesh.export(target_path)

    def augment_mesh(self, mesh):

        mesh = trimesh.util.concatenate(mesh)

        vertices = mesh.vertices
        faces = mesh.faces
        # vertices = fps_points(vertices, downsample_points=2048)

        mesh = pymeshlab.Mesh(vertex_matrix=np.array(vertices, ),
                              face_matrix=np.array(faces, dtype=np.int32))

        ms = pymeshlab.MeshSet()
        ms.add_mesh(mesh, 'my_mesh')
        ms.meshing_remove_duplicate_faces()
        ms.meshing_repair_non_manifold_edges()
        ms.meshing_repair_non_manifold_vertices()
        ms.meshing_surface_subdivision_midpoint(iterations=self.pymeshlab_iter)
        current_mesh = ms.current_mesh()

        vertices = current_mesh.vertex_matrix()
        faces = current_mesh.face_matrix()
        vertices = torch.tensor(vertices, dtype=torch.float64).unsqueeze(0)

        # if self.downsample_points is not None:

        #     vertices = math_utils.fps_points(vertices, downsample_points=2048)

        return vertices, faces


source_mesh_dir = "source/assets/ycb/dexgrasp/raw_mesh"
target_mesh_dir = "source/assets/ycb/dexgrasp/raw_mesh_augmented"
mesh_aug = MeshAugmentation(source_mesh_dir, target_mesh_dir)
mesh_aug.apply()
