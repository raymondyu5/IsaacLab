import numpy as np
from scipy.spatial.transform import Rotation as R
import json
from tools.visualization_utils import *


def plot_camera(ax, transform_matrix, scale=0.2, color='blue'):
    # Extract the rotation and translation components
    R = transform_matrix[:3, :3]
    # t = transform_matrix[:3, 3]

    # Define camera coordinates
    camera_points = np.array([[0, 0, 0], [scale, scale, 2 * scale],
                              [-scale, scale, 2 * scale],
                              [-scale, -scale, 2 * scale],
                              [scale, -scale, 2 * scale]])

    # Transform the points to world coordinates
    camera_points = (R @ camera_points.T).T

    # # Plot the camera
    # ax.plot([t[0], camera_points[1, 0]], [t[1], camera_points[1, 1]],
    #         [t[2], camera_points[1, 2]],
    #         color=color)
    # ax.plot([t[0], camera_points[2, 0]], [t[1], camera_points[2, 1]],
    #         [t[2], camera_points[2, 2]],
    #         color=color)
    # ax.plot([t[0], camera_points[3, 0]], [t[1], camera_points[3, 1]],
    #         [t[2], camera_points[3, 2]],
    #         color=color)
    # ax.plot([t[0], camera_points[4, 0]], [t[1], camera_points[4, 1]],
    #         [t[2], camera_points[4, 2]],
    #         color=color)

    # Draw lines between the points to form the pyramid
    ax.plot([camera_points[1, 0], camera_points[2, 0]],
            [camera_points[1, 1], camera_points[2, 1]],
            [camera_points[1, 2], camera_points[2, 2]],
            color=color)
    ax.plot([camera_points[2, 0], camera_points[3, 0]],
            [camera_points[2, 1], camera_points[3, 1]],
            [camera_points[2, 2], camera_points[3, 2]],
            color=color)
    ax.plot([camera_points[3, 0], camera_points[4, 0]],
            [camera_points[3, 1], camera_points[4, 1]],
            [camera_points[3, 2], camera_points[4, 2]],
            color=color)
    ax.plot([camera_points[4, 0], camera_points[1, 0]],
            [camera_points[4, 1], camera_points[1, 1]],
            [camera_points[4, 2], camera_points[1, 2]],
            color=color)


# Load transformation matrices from JSON file
with open(
        'logs/static_gs/cat/raw/Isaac-Lift-DeformCube-Franka-Play-v0/transforms.json',
        'r') as f:
    data = json.load(f)

transforms_isaac = {}
for frame in data['frames']:

    index = int(frame["file_path"].split("/")[1].split(".")[0].split("_")[1])
    transforms_isaac[index] = np.array(frame['transform_matrix'])

with open(
        'logs/static_gs/cat/raw/Isaac-Lift-DeformCube-Franka-Play-v0/metashape/transforms.json',
        'r') as f:
    data = json.load(f)
transforms_metashape = {}
for frame in data['frames']:
    index = int(frame["file_path"].split("/")[1].split(".")[0].split("_")[1])
    transforms_metashape[index] = np.array(frame['transform_matrix'])
import trimesh

mesh = trimesh.load(
    '/media/lme/data4/weird/IsaacLab/source/assets/Plush/usd/rabbit/rabbit.obj'
)
for key in transforms_metashape.keys():
    isaac_matrix = transforms_isaac[key]
    metashape_matrix = transforms_metashape[key]

    # Extract the rotation matrices (3x3 submatrices)
    # R1 = np.linalg.inv(isaac_matrix[:3, :3])
    # R2 = np.linalg.inv(metashape_matrix[:3, :3])

    R1 = isaac_matrix[:3, :3]
    R2 = metashape_matrix[:3, :3]
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=10.0, origin=[0, 0, 0])
    isaac_color = np.zeros_like(mesh.vertices)
    isaac_color[:, 1] = 255

    metashsape_color = np.zeros_like(mesh.vertices)
    metashsape_color[:, 2] = 255

    # rotate_matrix = math_utils.matrix_from_quat(
    #     obtain_target_quat_from_multi_angles([0], [1.57])).numpy()
    # R1 = rotate_matrix @ R1
    print(isaac_matrix[:3, 3], metashape_matrix[:3, 3])

    isaac_mesh = vis_pc((R1 @ mesh.vertices.T).T + isaac_matrix[:3, 3],
                        isaac_color)
    metashape_mesh = vis_pc((R2 @ mesh.vertices.T).T + metashape_matrix[:3, 3],
                            metashsape_color)
    o3d.visualization.draw_geometries(
        [isaac_mesh, metashape_mesh, coordinate_frame])

    # rotation = R.from_matrix(R1[:3, :3])
    # euler_angles_isaac = rotation.as_euler('xyz',
    #                                        degrees=True).astype(np.uint8)
    # rotation = R.from_matrix(R2[:3, :3])
    # euler_angles_metashape = rotation.as_euler('xyz',
    #                                            degrees=True).astype(np.uint8)
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # plot_camera(ax, R1, color=f'blue')
    # plot_camera(ax, R2, color=f'red')

    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # plt.show()
    # import pdb
    # pdb.set_trace()

    # Calculate the relative rotation matrix
    R_diff = R1 @ R2.T

    # Convert rotation matrix to rotation vector (axis-angle representation)
    rotation = R.from_matrix(R_diff)
    angle = rotation.magnitude()  # Get the rotation angle in radians
    axis = rotation.as_rotvec()  # Get the rotation axis

    angle_degrees = np.degrees(angle)
    print('=========================')
    # print(key, rotation.as_matrix())
    print(f"Rotation Angle Difference: {angle_degrees} degrees")
    print(f"Rotation Axis: {axis}")
    # import pdb
    # pdb.set_trace()
