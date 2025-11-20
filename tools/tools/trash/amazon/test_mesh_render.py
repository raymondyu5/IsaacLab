import numpy as np
import trimesh
import torch
import isaaclab.utils.math as math_utils
import open3d as o3d
import pymeshlab
from torch_cluster import fps


def fps_points(point_clouds, downsample_points=1024):
    # torch cluster fps

    point_clouds = point_clouds.reshape(-1, *point_clouds.shape[-2:])

    batch_size, num_points, num_dims = point_clouds.shape
    flattened_points = point_clouds.view(-1, num_dims)  # Shape: (71*10000, 3)

    # Create a batch tensor to indicate each point cloud's batch index
    batch = torch.repeat_interleave(torch.arange(batch_size),
                                    num_points).to(flattened_points.device)
    ratio = downsample_points / num_points

    # Apply farthest point sampling
    sampled_idx = fps(point_clouds[:, :, :3].reshape(-1, 3),
                      batch,
                      ratio=ratio,
                      batch_size=batch_size)
    sampled_points = flattened_points[sampled_idx]
    sampled_points_per_cloud = sampled_points.size(0) // batch_size
    output = sampled_points.view(batch_size, sampled_points_per_cloud,
                                 num_dims)

    return output


pose_data = np.load("logs/mesh_location_0.npz",
                    allow_pickle=True)["arr_0"].item()

link_names = list(pose_data.keys())
mesh_dict = []
for name in link_names:
    split_name = name.split("_")[1:]

    if split_name[-1].isdigit():
        split_name = split_name[:-1]  # Remove the last element
    mesh_name = "_".join(split_name)

    if mesh_name == '':
        continue
    if mesh_name is not None:
        if "palm" in name:
            mesh_name = name
        if "thumb" in name:

            mesh_name = "_".join(name.split("_")[1:])
        try:

            if "link" in name:
                mesh_name = "_".join(name.split("_")[1:])

                mesh = trimesh.load(
                    f"/home/ensu/Documents/weird/IsaacLab_assets/assets/robot/xarm7/raw_mesh/{mesh_name}.obj"
                )
                # mesh = trimesh.load(
                #     f"/home/ensu/Documents/weird/IsaacLab/logs/1743390994/raw_mesh/{mesh_name}.obj"
                # )
                # print(name, mesh_name)

            else:

                mesh = trimesh.load(
                    f"/home/ensu/Documents/weird/IsaacLab_assets/assets/robot/leap_hand_v2/raw_mesh/{mesh_name}.obj"
                )
                # mesh = trimesh.load(
                #     f"/home/ensu/Documents/weird/IsaacLab/logs/1743390994/raw_mesh/{mesh_name}.obj"
                # )

            # vertices = torch.as_tensor(mesh.vertices,   dtype=torch.float64) #.unsqueeze(0)

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
            ms.meshing_surface_subdivision_midpoint(iterations=5)
            current_mesh = ms.current_mesh()

            vertices = current_mesh.vertex_matrix()
            vertices = torch.tensor(vertices, dtype=torch.float64).unsqueeze(0)

            # vertices = fps_points(vertices, downsample_points=2048)

        except:
            continue
        pose = torch.as_tensor(pose_data[name], dtype=torch.float64)

        transformed_mvertices = math_utils.transform_points(
            vertices, pose[:, :3], pose[:, 3:7])
        mesh_dict.append(transformed_mvertices)
import pdb
from tools.visualization_utils import vis_pc, visualize_pcd

mesh_dict = torch.cat(mesh_dict, dim=1)

pc_list = vis_pc(mesh_dict.cpu().numpy()[0])
o3d.visualization.draw_geometries([pc_list])
pdb.set_trace()
