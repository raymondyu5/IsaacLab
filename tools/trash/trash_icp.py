import os
import numpy as np
# import open3d as o3d
import torch
from scripts.workflows.sysID.ASID.tool.icp import iterative_closest_point
from pytorch3d.structures import Pointclouds


def transform_pc(pc, extrinsic_matrix):
    """
    Transform point cloud using the given extrinsic matrix.
    """
    T_A_to_B = extrinsic_matrix
    R = T_A_to_B[:3, :3]
    t = T_A_to_B[:3, 3]

    # Compute inverse rotation and translation
    R_inv = R.T
    t_inv = -R_inv @ t

    # Assemble inverse transformation matrix
    T_B_to_A = np.eye(4)
    T_B_to_A[:3, :3] = R_inv
    T_B_to_A[:3, 3] = t_inv
    pc_homogeneous = np.hstack((pc[:, :3], np.ones((pc.shape[0], 1))))

    # Apply transformation
    transformed_pc_homogeneous = np.dot(pc_homogeneous, T_B_to_A.T)
    transformed_pc = transformed_pc_homogeneous[:, :3]

    # Normalize color values
    pc[:, 3:] /= 255

    # Filter points based on specific range
    pc_color = np.concatenate([transformed_pc, pc[:, 3:]], axis=1)
    transformed_pc_color = np.array([
        (x, y, z, r, g, b) for index, (x, y, z, r, g, b) in enumerate(pc_color)
        if -0.4 <= x <= 0.4 and -0.4 <= y <= 0.5 and -1 <= z <= -0.05
    ])

    # Create Open3D point cloud
    camera_o3d = o3d.geometry.PointCloud()
    camera_o3d.points = o3d.utility.Vector3dVector(transformed_pc_color[:, :3])
    camera_o3d.colors = o3d.utility.Vector3dVector(transformed_pc_color[:, 3:])

    return camera_o3d


def read_source_points():
    """
    Read and transform source points from files.
    """
    caliberation_files = os.listdir(calibration_dir)

    pc_list, transformed_pc = [], []

    for cali_file in caliberation_files:
        # Load calibration and point cloud data
        cali_data = np.load(os.path.join(calibration_dir, cali_file),
                            allow_pickle=True)
        rgbd_data = np.load(os.path.join(rgbd_data_dir, cali_file),
                            allow_pickle=True)

        # Extract matrices and images
        extrinsic_matrix = cali_data["extrinsic_matrix"]
        depth_image = rgbd_data["depth_image"]

        # Transform point cloud
        color_pc = transform_pc(depth_image, extrinsic_matrix)
        pc_list.append(color_pc)
        transformed_pc.append(
            np.concatenate(
                [np.asarray(color_pc.points),
                 np.asarray(color_pc.colors)],
                axis=1))

    return pc_list, np.concatenate(transformed_pc, axis=0)


def numpy_to_pointcloud(point_cloud_np):
    """
    Convert a numpy point cloud to a PyTorch3D Pointclouds object.
    """
    point_cloud_tensor = torch.tensor(point_cloud_np, dtype=torch.float32)
    return Pointclouds(points=[point_cloud_tensor])


def perform_icp(source_pc_np,
                target_pc_np,
                max_iterations=100,
                tolerance=1e-6,
                device=None):
    """
    Perform Iterative Closest Point (ICP) registration between two point clouds.
    """
    source_pc = numpy_to_pointcloud(source_pc_np)
    target_pc = numpy_to_pointcloud(target_pc_np)

    device = device or torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")
    source_pc, target_pc = source_pc.to(device), target_pc.to(device)

    # Perform ICP
    icp_transform = iterative_closest_point(source_pc,
                                            target_pc,
                                            max_iterations=max_iterations)

    # Extract transformation matrix
    rotation_matrix = icp_transform.RTs.R[0].detach().cpu().numpy()
    translation_vector = icp_transform.RTs.T[0].detach().cpu().numpy()
    scale = icp_transform.RTs.s[0].detach().cpu().item()

    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix * scale
    transformation_matrix[:3, 3] = translation_vector

    # Transform source point cloud
    source_pc_np = source_pc.points_padded()[0].detach().cpu().numpy()
    source_pc_homogeneous = np.hstack(
        [source_pc_np, np.ones((source_pc_np.shape[0], 1))])
    transformed_source_pc_homogeneous = source_pc_homogeneous @ transformation_matrix.T
    transformed_source_pc = transformed_source_pc_homogeneous[:, :3]

    return transformation_matrix, transformed_source_pc


def perform_icp_multiple_iterations(source_pc_np,
                                    target_pc_np,
                                    num_iterations=10,
                                    max_iterations_per_icp=100,
                                    tolerance=1e-6,
                                    device=None):
    """
    Perform ICP for multiple iterations, updating the source point cloud after each iteration.
    """
    final_transformation = np.eye(4)
    for i in range(num_iterations):
        print(f"ICP iteration {i + 1}/{num_iterations}")
        transformation_matrix, transformed_source_pc = perform_icp(
            source_pc_np,
            target_pc_np,
            max_iterations=max_iterations_per_icp,
            tolerance=tolerance,
            device=device)

        # Update source point cloud for the next iteration
        source_pc_np = transformed_source_pc

        # Combine the transformations
        final_transformation = final_transformation @ transformation_matrix

    return final_transformation, transformed_source_pc


def numpy_to_open3d_point_cloud_with_color(point_cloud_np):
    """
    Convert a numpy array (N, 6) to an Open3D PointCloud object with color.
    """
    point_cloud_o3d = o3d.geometry.PointCloud()
    points, colors = point_cloud_np[:, :3], point_cloud_np[:, 3:] / 255.0
    point_cloud_o3d.points = o3d.utility.Vector3dVector(points)
    point_cloud_o3d.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud_o3d


def visualize_point_clouds_with_color(source_pc_np, target_pc_np,
                                      transformed_source_pc_np):
    """
    Visualize the source, target, and transformed point clouds using Open3D.
    """
    source_pc_o3d = numpy_to_open3d_point_cloud_with_color(source_pc_np)
    target_pc_o3d = numpy_to_open3d_point_cloud_with_color(target_pc_np)
    transformed_source_pc_o3d = numpy_to_open3d_point_cloud_with_color(
        transformed_source_pc_np)

    o3d.visualization.draw_geometries(
        [source_pc_o3d, target_pc_o3d, transformed_source_pc_o3d],
        window_name="ICP Result with Color")


# Define paths outside the function
calibration_dir = '/media/lme/data4/weird/droid/trash/caliberation'
rgbd_data_dir = '/media/lme/data4/weird/droid/trash/pc'
object_mesh_path = "/media/lme/data4/weird/IsaacLab/source/assets/Plush/usd/cat/cat.obj"

# read target point cloud
pc_list, transformed_pc = read_source_points()
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.1, origin=[0.0, 0.0, 0.0])

indices = np.random.choice(transformed_pc.shape[0], 2048 * 10, replace=False)
target_pc = transformed_pc[indices, :]

target_pc[:, 2] *= -1
target_pc[:, 3:] *= 1

target_pc[:, :3] -= np.mean(target_pc[:, :3], axis=0)
target_pc[:, 0] += 0.3
# read source point cloud
# Load the mesh
object_mesh = o3d.io.read_triangle_mesh(object_mesh_path)
source_pc_np = np.array(object_mesh.vertices)
target_pc = np.copy(source_pc_np)

for i in range(10):
    random_angle_z = np.pi / 10 * i
    # Create the rotation matrix for rotation around the z-axis
    rotation_matrix_z = np.array(
        [[np.cos(random_angle_z), -np.sin(random_angle_z), 0],
         [np.sin(random_angle_z),
          np.cos(random_angle_z), 0], [0, 0, 1]])

    # Apply the rotation to the source point cloud (vertices of the mesh)
    source_pc_np_rotated = np.dot(source_pc_np, rotation_matrix_z.T)

    # Add color to the point cloud (all red for your case)
    source_pc_color = np.ones_like(source_pc_np_rotated) * 0
    source_pc_color[:, 0] = 255  # Red color

    target_pc_color = np.ones_like(source_pc_np_rotated) * 0
    target_pc_color[:, 1] = 255  # Red color

    # Combine the rotated point cloud with the color information
    target_pc = np.concatenate([target_pc, target_pc_color], axis=1)

    # Combine the rotated point cloud with the color information
    source_pc_np_rotated = np.concatenate(
        [source_pc_np_rotated, source_pc_color], axis=1)
    source_pc_np_rotated[:, :3] -= np.mean(source_pc_np_rotated[:, :3], axis=0)

    # Perform ICP with multiple iterations
    final_transformation_matrix, final_transformed_source_pc = perform_icp_multiple_iterations(
        source_pc_np_rotated[..., :3], target_pc[..., :3], num_iterations=1)

    transform_pc_color = np.zeros_like(source_pc_np[:, :3])
    transform_pc_color[:, 2] = 255

    # Visualize the results
    visualize_point_clouds_with_color(
        source_pc_np_rotated[..., :6], target_pc[..., :6],
        np.concatenate(
            [final_transformed_source_pc[..., :3], transform_pc_color],
            axis=1))
