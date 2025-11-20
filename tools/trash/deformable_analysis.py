import sys

sys.path.append(".")
from tools.visualization_utils import *
import numpy as np
# import open3d as o3d
import trimesh

pc_list = []
for index, reso in enumerate([4, 8, 16, 32, 64]):
    mesh = o3d_fem_mesh(f"logs/rabbit/deform_fem_{reso}/0/5.obj",
                        translate=[0.2 * index, 0, 0])
    raw_vertices = trimesh.load(
        f"source/assets/Plush/usd/rabbit/rabbit.obj").vertices / 100 * 0.6

    translation = np.array([
        0.47 - 0.2 * index, -0.0,
        (np.max(raw_vertices, 0)[1] - np.min(raw_vertices, 0)[1]) / 2
    ])
    quat = obtain_target_quat_from_multi_angles([1, 2, 0, 1],
                                                [-1.57, 1.57, -1.57, 1.57])

    transformed_vertices = transform_vertices(raw_vertices, quat, translation)
    pc = vis_pc(transformed_vertices)
    pc_list.append(mesh)
    pc_list.append(pc)
    # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

visualize_pcd(pc_list,
              translation=(-0.1, 0, 0),
              rotation_axis=[1],
              rotation_angles=[0])
# o3d.visualization.draw_geometries([mesh,pc], mesh_show_wireframe=True)
