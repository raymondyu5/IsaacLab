import h5py
import cv2
import os
import isaaclab.utils.math as math_utils
import numpy as np
from pathlib import Path
import json
import torch
import math
from scipy.spatial.transform import Rotation as R


def convert_camera_frame_orientation_convention(
    orientation,
    origin,
    target,
) -> torch.Tensor:
    r"""Converts a quaternion representing a rotation from one convention to another.

    In USD, the camera follows the ``"opengl"`` convention. Thus, it is always in **Y up** convention.
    This means that the camera is looking down the -Z axis with the +Y axis pointing up , and +X axis pointing right.
    However, in ROS, the camera is looking down the +Z axis with the +Y axis pointing down, and +X axis pointing right.
    Thus, the camera needs to be rotated by :math:`180^{\circ}` around the X axis to follow the ROS convention.

    .. math::

        T_{ROS} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & -1 & 0 & 0 \\ 0 & 0 & -1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix} T_{USD}

    On the other hand, the typical world coordinate system is with +X pointing forward, +Y pointing left,
    and +Z pointing up. The camera can also be set in this convention by rotating the camera by :math:`90^{\circ}`
    around the X axis and :math:`-90^{\circ}` around the Y axis.

    .. math::

        T_{WORLD} = \begin{bmatrix} 0 & 0 & -1 & 0 \\ -1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix} T_{USD}

    Thus, based on their application, cameras follow different conventions for their orientation. This function
    converts a quaternion from one convention to another.

    Possible conventions are:

    - :obj:`"opengl"` - forward axis: -Z - up axis +Y - Offset is applied in the OpenGL (Usd.Camera) convention
    - :obj:`"ros"`    - forward axis: +Z - up axis -Y - Offset is applied in the ROS convention
    - :obj:`"world"`  - forward axis: +X - up axis +Z - Offset is applied in the World Frame convention

    Args:
        orientation: Quaternion of form `(w, x, y, z)` with shape (..., 4) in source convention
        origin: Convention to convert to. Defaults to "ros".
        target: Convention to convert from. Defaults to "opengl".

    Returns:
        Quaternion of form `(w, x, y, z)` with shape (..., 4) in target convention
    """
    if target == origin:
        return orientation.clone()

    # -- unify input type
    if origin == "ros":
        # convert from ros to opengl convention
        rotm = math_utils.matrix_from_quat(orientation)
        rotm[:, :, 2] = -rotm[:, :, 2]
        rotm[:, :, 1] = -rotm[:, :, 1]
        # convert to opengl convention
        quat_gl = math_utils.quat_from_matrix(rotm)
    elif origin == "world":
        # convert from world (x forward and z up) to opengl convention
        rotm = math_utils.matrix_from_quat(orientation)
        rotm = torch.matmul(
            rotm,
            math_utils.matrix_from_euler(
                torch.tensor([math.pi / 2, -math.pi / 2, 0],
                             device=orientation.device), "XYZ"),
        )
        # convert to isaac-sim convention
        quat_gl = math_utils.quat_from_matrix(rotm)
    else:
        quat_gl = orientation

    # -- convert to target convention
    if target == "ros":
        # convert from opengl to ros convention
        rotm = math_utils.matrix_from_quat(quat_gl)
        rotm[:, :, 2] = -rotm[:, :, 2]
        rotm[:, :, 1] = -rotm[:, :, 1]
        return math_utils.quat_from_matrix(rotm)
    elif target == "world":
        # convert from opengl to world (x forward and z up) convention
        rotm = math_utils.matrix_from_quat(quat_gl)
        rotm = torch.matmul(
            rotm,
            math_utils.matrix_from_euler(
                torch.tensor([math.pi / 2, -math.pi / 2, 0],
                             device=orientation.device), "XYZ").T,
        )
        return math_utils.quat_from_matrix(rotm)
    else:
        return quat_gl.clone()


def obtain_target_quat_from_multi_angles(axis, angles):
    quat_list = []
    for index, cam_axis in enumerate(axis):
        euler_xyz = torch.zeros(3)
        euler_xyz[cam_axis] = angles[index]
        quat_list.append(
            math_utils.quat_from_euler_xyz(euler_xyz[0], euler_xyz[1],
                                           euler_xyz[2]))
    if len(quat_list) == 1:
        return quat_list[0]
    else:
        target_quat = quat_list[0]
        for index in range(len(quat_list) - 1):

            target_quat = math_utils.quat_mul(quat_list[index + 1],
                                              target_quat)
        return target_quat


dir_name = "/media/lme/data4/weird/IsaacLab/logs/static_gs/cat/raw/Isaac-Lift-DeformCube-Franka-Play-v0"
type = "cat"
keep_original_world_coordinate = False
os.makedirs(f"{dir_name}/raw_imgs", exist_ok=True)
with h5py.File(f"{dir_name}/{type}.hdf5", 'r') as file:

    rgb_data = file["data"]["demo_0"]["obs"]["rgb"][:, :, :, :, :3]
    extrinsic_params = file["data"]["demo_0"]["obs"]["extrinsic_params"][
        :,
    ]
    intrinsic_params = file["data"]["demo_0"]["obs"]["intrinsic_params"][
        :,
    ]

    rgb_dataset = rgb_data.reshape(-1, rgb_data.shape[2], rgb_data.shape[3], 3)

    extrinsic_dataset = extrinsic_params.reshape(-1, extrinsic_params.shape[2],
                                                 extrinsic_params.shape[3])

    intrinsic_dataset = intrinsic_params.reshape(-1, intrinsic_params.shape[2],
                                                 intrinsic_params.shape[3])
    frames = []
    out = {}

    intrinsic = intrinsic_dataset[0]
    # import pdb
    # pdb.set_trace()
    out["w"] = rgb_dataset[0].shape[1]
    out["h"] = rgb_dataset[0].shape[0]
    out["fl_x"] = float(intrinsic[0][0])
    out["fl_y"] = float(intrinsic[0][0])
    out["cx"] = float(intrinsic[0][2])
    out["cy"] = float(intrinsic[1][2])
    out["k1"] = float(0.0)
    out["k2"] = float(0.0)
    out["p1"] = float(0.0)
    out["p2"] = float(0.0)
    out["camera_model"] = "OPENCV"
    for img_idx in range(rgb_dataset.shape[0]):
        cv2.imwrite(
            f"{dir_name}/raw_imgs/frame_{img_idx:05d}.jpg",
            rgb_dataset[img_idx][..., ::-1],
        )
        rotate_matrix = np.eye(3)

        # rotate_matrix[:3, :3] = math_utils.matrix_from_quat(
        #     obtain_target_quat_from_multi_angles([0, 1, 0],
        #                            [1.57, 1.57, 3.14 + 1.57])).numpy()
        w2c = extrinsic_dataset[img_idx]
        # Convert rotation matrix to a Rotation object
        rotation = R.from_matrix(w2c[:3, :3])

        # Convert the Rotation object to Euler angles
        # The argument 'xyz' defines the axes of rotation, you can change it according to your need.
        euler_angles = rotation.as_euler('xyz', degrees=True).astype(np.uint8)
        print()
        import pdb
        pdb.set_trace()
    #     # w2c[:3, :3] = rotate_matrix @ extrinsic_dataset[img_idx][:3, :3]
    #     c2w = np.linalg.inv(w2c)

    #     # c2w[0:3, 1:3] *= -1

    #     # # if not keep_original_world_coordinate:
    #     # c2w = c2w[np.array([0, 2, 1, 3]), :]
    #     # c2w[2, :] *= -1

    #     # c2w[:3, :3] = rotation_opengl
    #     name = f"raw_imgs/frame_{img_idx:05d}.jpg"
    #     name = Path(name)

    #     frame = {
    #         "file_path": name.as_posix(),
    #         "transform_matrix": c2w.tolist(),
    #         "colmap_im_id": img_idx,
    #     }

    #     frames.append(frame)
    # out["frames"] = frames
    # applied_transform = None

    # with open(Path(dir_name) / "transforms.json", "w", encoding="utf-8") as f:
    #     json.dump(out, f, indent=4)
