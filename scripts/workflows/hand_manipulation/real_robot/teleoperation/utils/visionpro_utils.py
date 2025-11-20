import psutil
import os
import signal

import numpy as np
from pytransform3d import rotations

from pytransform3d import rotations, coordinates

MANO2ROBOT = np.array([
    [-1, 0, 0],
    [0, 0, -1],
    [0, -1, 0],
])

OPERATOR2MANO_RIGHT = np.array([
    [0, 0, -1],
    [-1, 0, 0],
    [0, 1, 0],
])

OPERATOR2MANO_LEFT = np.array([
    [0, 0, -1],
    [1, 0, 0],
    [0, -1, 0],
])

OPERATOR2AVP_RIGHT = OPERATOR2MANO_RIGHT

OPERATOR2AVP_LEFT = np.array([
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
])

import socket


def send_command_to_vision_pro(ip_address: str, port: int, command: str):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((ip_address, port))
            s.sendall(command.encode('utf-8'))
            # print(f"✅ Sent command to Vision Pro: {command}")
    except Exception as e:
        print(f"❌ Failed to send command: {e}")


def kill_process_using_port(port):
    for proc in psutil.process_iter(['pid', 'name', 'connections']):
        try:
            for conn in proc.info['connections']:
                if conn.laddr.port == port:
                    print(f"🔪 Killing process {proc.pid} using port {port}")
                    os.kill(proc.pid, signal.SIGKILL)
                    break
        except Exception:
            continue


def three_mat_mul(left_rot: np.ndarray, mat: np.ndarray,
                  right_rot: np.ndarray):
    result = np.eye(4)
    rotation = left_rot @ mat[:3, :3] @ right_rot
    pos = left_rot @ mat[:3, 3]
    result[:3, :3] = rotation
    result[:3, 3] = pos
    return result


def rotate_head(R, degrees=-90):
    # Convert degrees to radians
    theta = np.radians(degrees)
    # Create the rotation matrix for rotating around the x-axis
    R_x = np.array([
        [1, 0, 0, 0],
        [0, np.cos(theta), -np.sin(theta), 0],
        [0, np.sin(theta), np.cos(theta), 0],
        [0, 0, 0, 1],
    ])
    R_rotated = R @ R_x
    return R_rotated


# def two_mat_batch_mul(batch_mat: np.ndarray, left_rot: np.ndarray):
#     result = np.tile(np.eye(4), [batch_mat.shape[0], 1, 1])
#     result[:, :3, :3] = np.matmul(left_rot[None, ...], batch_mat[:, :3, :3])
#     result[:, :3, 3] = batch_mat[:, :3, 3] @ left_rot.T
#     return result


def two_mat_batch_mul(batch_mat: np.ndarray, left_rot: np.ndarray):
    batch_size = batch_mat.shape[0]

    # Compute new rotation: left_rot @ batch_rot (broadcasted)
    rot_new = left_rot @ batch_mat[:, :3, :3]  # shape: (B, 3, 3)

    # Compute new translation: batch_trans @ left_rot.T
    trans_new = batch_mat[:, :3, 3] @ left_rot.T  # shape: (B, 3)

    # Construct full 4x4 matrices efficiently
    result = np.zeros((batch_size, 4, 4), dtype=batch_mat.dtype)
    result[:, :3, :3] = rot_new
    result[:, :3, 3] = trans_new
    result[:, 3, 3] = 1.0  # Homogeneous coordinate

    return result


def joint_avp2hand(finger_mat: np.ndarray):
    finger_index = np.array([
        0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23,
        24
    ])
    finger_mat = finger_mat[finger_index]
    return finger_mat


def project_average_rotation(quat_list: np.ndarray):
    gravity_dir = np.array([0, 0, -1])

    last_quat = quat_list[-1, :]
    last_mat = rotations.matrix_from_quaternion(last_quat)
    gravity_quantity = gravity_dir @ last_mat  # (3, )
    max_gravity_axis = np.argmax(np.abs(gravity_quantity))
    same_direction = gravity_quantity[max_gravity_axis] > 0

    next_axis = (max_gravity_axis + 1) % 3
    next_next_axis = (max_gravity_axis + 2) % 3
    angles = []
    for i in range(quat_list.shape[0]):
        next_dir = rotations.matrix_from_quaternion(quat_list[i])[:3,
                                                                  next_axis]
        next_dir[2] = 0  # Projection to non gravity direction
        next_dir_angle = coordinates.spherical_from_cartesian(next_dir)[2]
        angles.append(next_dir_angle)

    angle = np.mean(angles)
    final_mat = np.zeros([3, 3])
    final_mat[:3, max_gravity_axis] = gravity_dir * same_direction
    final_mat[:3, next_axis] = [np.cos(angle), np.sin(angle), 0]
    final_mat[:3, next_next_axis] = np.cross(final_mat[:3, max_gravity_axis],
                                             final_mat[:3, next_axis])

    return final_mat


class LPFilter:

    def __init__(self, alpha):
        self.alpha = alpha
        self.y = None
        self.is_init = False

    def next(self, x):
        if not self.is_init:
            self.y = x
            self.is_init = True
            return self.y.copy()
        self.y = self.y + self.alpha * (x - self.y)
        return self.y.copy()

    def reset(self):
        self.y = None
        self.is_init = False


class LPRotationFilter:

    def __init__(self, alpha):
        self.alpha = alpha
        self.is_init = False

        self.y = None

    def next(self, x: np.ndarray):
        assert x.shape == (4, )

        if not self.is_init:
            self.y = x
            self.is_init = True
            return self.y.copy()

        self.y = rotations.quaternion_slerp(self.y,
                                            x,
                                            self.alpha,
                                            shortest_path=True)
        return self.y.copy()

    def reset(self):
        self.y = None
        self.is_init = False


from curobo.util_file import (
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)

from curobo.types.base import TensorDeviceType

from curobo.types.robot import JointState, RobotConfig
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.types.math import Pose
import torch


class IKPlanner:

    def __init__(
        self,
        env=None,
        device=None,
        robot_file="franka.yml",
        world_file="collision_table.yml",
        only_paths=None,
        reference_prim_path=None,
        ignore_substring=None,
    ):
        robot_file = join_path(get_robot_configs_path(), robot_file)
        world_file = join_path(get_world_configs_path(), world_file)

        self.device = device
        self.tensor_args = TensorDeviceType()

        robot_cfg = load_yaml(join_path(get_robot_configs_path(),
                                        robot_file))["robot_cfg"]
        robot_cfg = RobotConfig.from_dict(robot_cfg, self.tensor_args)

        n_obstacle_cuboids = 30
        n_obstacle_mesh = 10
        from curobo.geom.types import Cuboid, WorldConfig
        tensor_args = TensorDeviceType()
        ik_config = IKSolverConfig.load_from_robot_config(
            robot_cfg,
            None,
            rotation_threshold=0.05,
            position_threshold=0.005,
            num_seeds=20,
            self_collision_check=False,
            self_collision_opt=False,
            tensor_args=tensor_args,
            use_cuda_graph=True,
        )
        self.ik_solver = IKSolver(ik_config)

        if env is not None:
            from curobo.util.usd_helper import UsdHelper

            self.usd_help = UsdHelper()
            self.usd_help.load_stage(env.scene.stage)
            from curobo.geom.types import Cuboid, WorldConfig
            self.world = WorldConfig()
            self.usd_help.add_world_to_stage(self.world, base_frame="/World")

            self.only_paths = only_paths
            self.reference_prim_path = reference_prim_path
            self.ignore_substring = ignore_substring

    def plan_motion(
        self,
        target_position,
        target_quat,
        # default_joints,
    ):

        goal = Pose(target_position, target_quat)

        result = self.ik_solver.solve_batch(goal)

        torch.cuda.empty_cache()
        return result.js_solution.position
