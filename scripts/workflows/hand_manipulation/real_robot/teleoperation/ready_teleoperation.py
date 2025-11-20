import torch
import isaaclab.utils.math as math_utils
from isaaclab.envs import mdp
from isaaclab.sensors.camera.utils import obtain_target_quat_from_multi_angles
import numpy as np

init_hand_poses = {
    "face_down": [0.300, -0.000, 0.400, 0.0, 9.2460e-01, -3.8094e-01,
                  0.0],  # 180° around X
    "face_left": [0.300, -0.20, 0.400, -0.6537, 0.6537, -0.2693, -0.2693],
    # "face_right": [0.500, -0.000, 0.500, 0.7071, 0.0, -0.7071, 0.0],
    # "face_forward": [0.500, -0.000, 0.500, 0.7071, 0.7071, 0.0, 0.0],
    # "face_backward": [0.500, -0.000, 0.500, 0.7071, -0.7071, 0.0, 0.0],
}


class LPFilter:

    def __init__(self, alpha):
        self.alpha = alpha
        self.y = None
        self.is_init = False

    def next(self, x):
        if not self.is_init:
            self.y = x
            self.is_init = True
            return self.y.clone()
        self.y = self.y + self.alpha * (x - self.y)
        return self.y.clone()

    def reset(self):
        self.y = None
        self.is_init = False


class LPRotationFilter:

    def __init__(self, alpha):
        self.alpha = alpha
        self.is_init = False

        self.y = None

    def next(self, x: torch.Tensor):
        assert x.shape == (4, )

        if not self.is_init:
            self.y = x
            self.is_init = True
            return self.y.clone()

        self.y = math_utils.quat_slerp(
            self.y,
            x,
            self.alpha,
        )
        return self.y.clone()

    def reset(self):
        self.y = None
        self.is_init = False


def project_average_rotation(quat_list: torch.Tensor) -> torch.Tensor:

    gravity_dir = torch.as_tensor([0, 0,
                                   -1]).to(quat_list.device).to(torch.float32)

    last_quat = quat_list[-1, :]
    last_mat = math_utils.matrix_from_quat(last_quat)
    gravity_quantity = gravity_dir @ last_mat.to(torch.float32)  # (3, )
    max_gravity_axis = torch.argmax(torch.abs(gravity_quantity))
    same_direction = gravity_quantity[max_gravity_axis] > 0

    next_axis = (max_gravity_axis + 1) % 3
    next_next_axis = (max_gravity_axis + 2) % 3
    angles = []
    next_dir = math_utils.matrix_from_quat(quat_list)[:, :3, next_axis]
    next_dir[:, 2] = 0
    from pytransform3d import rotations, coordinates

    for i in range(next_dir.shape[0]):
        angles.append(
            coordinates.spherical_from_cartesian(next_dir[i].cpu().numpy())[2])

    angle = np.mean(angles)
    final_mat = torch.zeros((3, 3)).to(quat_list.device).to(torch.float32)
    final_mat[:3, max_gravity_axis] = gravity_dir * same_direction
    final_mat[0, next_axis] = np.cos(angle)
    final_mat[1, next_axis] = np.sin(angle)

    final_mat[:3,
              next_next_axis] = torch.cross(final_mat[:3, max_gravity_axis],
                                            final_mat[:3, next_axis])
    return final_mat


class ReadyTeleoperation:

    def __init__(self, env, teleop_interface, args_cli, env_cfg, teleop_config,
                 init_hand_dir):
        self.use_relative_pose = True if "Rel" in args_cli.task else False
        self.teleop_interface = teleop_interface
        self.env = env
        self.args_cli = args_cli
        self.env_cfg = env_cfg
        self.add_left_hand = args_cli.add_left_hand
        self.add_right_hand = args_cli.add_right_hand
        self.hand_side = "right" if self.add_right_hand else "left"
        self.device = env.device
        self.ready_for_teleoperation = False
        self.teleop_counter = 0

        # init calibration settings
        self.last_wrist_ee_pose = None
        self.last_wrist_ee_quat = None
        self.ready_for_teleoperation_counter = 0
        self.init_pose_buffer = []
        self.init_fingier_pose_buffer = []
        self.teleop_config = teleop_config
        self.pose_factor = self.teleop_config.get("pose_factor", 1.0)
        low_pass_smoothing_wrist = self.teleop_config.get(
            "low_pass_smoothing_wrist", 0.1)
        self.use_motion_filter = self.teleop_config.get(
            "use_motion_filter", False)  # for debugging
        if self.use_motion_filter:
            self.wrist_pos_filter = LPFilter(alpha=low_pass_smoothing_wrist)
            self.wrist_rot_filter = LPRotationFilter(
                alpha=low_pass_smoothing_wrist)

        # self.left_hand_quat_offset = torch.tensor([
        #     .0000, 0.7070, -0.7070, 0.0000
        # ]).to(device=self.env.device).unsqueeze(0).repeat_interleave(
        #     self.env.num_envs, dim=0)
        self.hand_quat_offset = torch.tensor([
            .0000, 0.7070, 0.7070, 0.0000
        ]).to(device=self.env.device).unsqueeze(0).repeat_interleave(
            self.env.num_envs, dim=0)

        self.hand_pos_offset = torch.tensor(
            [0.0, 0.0,
             0.0]).to(device=self.env.device).unsqueeze(0).repeat_interleave(
                 self.env.num_envs, dim=0)

        self.teleop_interface._retargeters[
            0].right_hand_pos_offset = self.hand_pos_offset.cpu().numpy()[0]
        self.num_hand_joint = self.env_cfg["params"]["num_hand_joints"]
        self.init_hand_dir = init_hand_dir
        self.init_setting()
        self.num_frame = 0

    def init_setting(self):

        from scripts.workflows.hand_manipulation.env.teleop_env.motion_planner_env import ArmMotionPlannerEnv

        self.arm_motion_env = ArmMotionPlannerEnv(
            self.env,
            self.args_cli,
            self.env_cfg,
        )

        self.init_ee_pose = torch.as_tensor(
            init_hand_poses[self.init_hand_dir]).to(self.device).unsqueeze(0)

        self.init_arm_qpos = self.arm_motion_env.ik_plan_motion(
            self.init_ee_pose).squeeze(1)
        self.env_ids = torch.arange(self.env.num_envs).to(self.device)

        self.init_robot_qpos = torch.cat([
            self.init_arm_qpos,
            torch.zeros((1, self.num_hand_joint)).to(self.device)
        ],
                                         dim=1)
        init_pose = torch.cat([
            self.init_ee_pose,
            torch.zeros(1, self.num_hand_joint).to(self.device)
        ],
                              dim=1).repeat_interleave(self.env.num_envs,
                                                       dim=0)
        if self.use_relative_pose:
            self.init_actions = torch.zeros(self.env.action_space.shape).to(
                self.env.device)
        else:
            self.init_actions = init_pose

        for i in range(30):

            self.env.scene[
                f"{self.hand_side}_hand"].root_physx_view.set_dof_positions(
                    self.init_robot_qpos, indices=self.env_ids)

            self.env.step(self.init_actions)

        self.palm2wrist, self.wrist2palm = self.init_delta_transform()
        self.env.scene[
            f"{self.hand_side}_hand"]._data.reset_joint_pos[:, :
                                                            7] = self.init_arm_qpos.reshape(
                                                                -1)

    def init_delta_transform(self, ):
        palm_state = self.env.scene[
            f"{self.hand_side}_palm_lower"]._data.root_state_w[:, :7]
        wrist_state = self.env.scene[
            f"{self.hand_side}_panda_link7"]._data.root_state_w[:, :7]
        palm2wrist = torch.cat(math_utils.subtract_frame_transforms(
            palm_state[:, :3], palm_state[:, 3:7], wrist_state[:, :3],
            wrist_state[:, 3:7]),
                               dim=1)
        wrist2palm = torch.cat(math_utils.subtract_frame_transforms(
            wrist_state[:, :3], wrist_state[:, 3:7], palm_state[:, :3],
            palm_state[:, 3:7]),
                               dim=1)

        return palm2wrist, wrist2palm

    def apply_offset(self, teleop_data):

        hand_pose = torch.zeros((self.env.num_envs, 7), device=self.device)
        # hand_pose[..., 3:7] = torch.as_tensor(teleop_data[3:7]).to(
        #     self.device).unsqueeze(0).repeat_interleave(self.env.num_envs,
        #                                                 dim=0)
        hand_pose[..., 3:7] = math_utils.quat_mul(
            torch.as_tensor(teleop_data[3:7]).to(
                self.device).unsqueeze(0).repeat_interleave(self.env.num_envs,
                                                            dim=0),
            self.hand_quat_offset)

        hand_pose[..., :3] = torch.as_tensor(teleop_data[:3]).to(
            self.device).unsqueeze(0).repeat_interleave(
                self.env.num_envs, dim=0) + self.hand_pos_offset
        wrist_ee_pose = torch.cat(math_utils.combine_frame_transforms(
            hand_pose[:, :3], hand_pose[:, 3:7], self.palm2wrist[:, :3],
            self.palm2wrist[:, 3:7]),
                                  dim=1)
        return wrist_ee_pose

    @staticmethod
    def _compute_hand_joint_angles(joints: np.ndarray):
        tip_index = np.array([4, 8, 12, 16, 20])
        palm_bone_index = np.array([1, 5, 9, 13, 17])
        root = joints[0:1, :]
        tips = joints[tip_index]
        palm_bone = joints[palm_bone_index]
        tip_vec = tips - root
        tip_vec = tip_vec / np.linalg.norm(tip_vec, axis=1, keepdims=True)
        palm_vec = palm_bone - root
        palm_vec = palm_vec / np.linalg.norm(palm_vec, axis=1, keepdims=True)
        angle = np.arccos(
            np.clip(np.sum(tip_vec * palm_vec, axis=1), -1.0, 1.0))
        return angle

    def _compute_init_frame(self,
                            init_frame_pos: torch.Tensor,
                            init_frame_quat: torch.Tensor,
                            init_finger_pose: torch.Tensor,
                            align_gravity_dir=True) -> tuple:
        num_data = init_frame_pos.shape[0]
        weight = ((torch.arange(num_data) + 1) /
                  torch.sum(torch.arange(num_data) + 1)).to(self.device)

        if align_gravity_dir:
            calibrated_wrist_rot = project_average_rotation(init_frame_quat)
        else:
            calibrated_wrist_rot = math_utils.matrix_from_quat(
                init_frame_quat[-1])

        calibrated_wrist_pos = torch.sum(weight[:, None] * init_frame_pos,
                                         dim=0).to(torch.float32)
        calibrated_finger_pose = torch.sum(weight[:, None] * init_finger_pose,
                                           dim=0).to(torch.float32)

        return calibrated_wrist_pos.reshape(
            3), calibrated_wrist_rot, calibrated_finger_pose

    def _compute_pose_in_init_frame(self, pos: torch.Tensor,
                                    quat: torch.Tensor,
                                    finger_pose: torch.Tensor):

        operator_rot = math_utils.matrix_from_quat(quat)
        local_operator_rot = self.calibrated_wrist_rot.T @ operator_rot
        local_operator_pos = pos - self.calibrated_wrist_pos
        local_operator_quat = math_utils.quat_from_matrix(local_operator_rot)
        local_operator_finger_pose = finger_pose - self.calibrated_finger_pose

        if self.use_motion_filter:

            local_operator_pos = self.wrist_pos_filter.next(local_operator_pos)
            local_operator_quat = self.wrist_rot_filter.next(
                local_operator_quat)

        return local_operator_pos, local_operator_quat, local_operator_finger_pose

    def evaluate_init_pose(self,
                           ee_pose: torch.Tensor,
                           joint_angles: np.ndarray,
                           dist_threshold=0.07,
                           angle_threshold=20):
        intructions = "Please place your hand in the initial position"

        cur_rest_ee_pose = ee_pose[:3]
        cur_rest_ee_quat = ee_pose[3:7]
        cur_ee_joint_angles = ee_pose[7:]

        if self.last_wrist_ee_pose is None:
            self.last_wrist_ee_pose = cur_rest_ee_pose.clone()
            self.last_wrist_ee_quat = cur_rest_ee_quat.clone()

        else:

            moving_dist = torch.linalg.norm(cur_rest_ee_pose -
                                            self.last_wrist_ee_pose,
                                            dim=-1).cpu().numpy()

            flat_threshold = (0.01, np.deg2rad(angle_threshold))
            finger_angles = self._compute_hand_joint_angles(joint_angles)

            hand_flat_spread = (flat_threshold[0] < np.mean(finger_angles) <
                                flat_threshold[1])
            not_moving = moving_dist < dist_threshold

            if not_moving and hand_flat_spread:

                self.ready_for_teleoperation_counter += 1
                self.init_pose_buffer.append(ee_pose.clone())
                self.init_fingier_pose_buffer.append(cur_ee_joint_angles)
            else:
                self.ready_for_teleoperation_counter = 0
                self.init_pose_buffer.clear()
                self.init_fingier_pose_buffer.clear()

            if not_moving and hand_flat_spread and self.ready_for_teleoperation_counter > 50:
                print(f"[INFO]: Ready for teleoperation")
                self.ready_for_teleoperation = True
                self.ready_for_teleoperation_counter = 0

                init_pose_buffer = torch.stack(self.init_pose_buffer)
                init_finger_pose_buffer = torch.stack(
                    self.init_fingier_pose_buffer)

                self.calibrated_wrist_pos, self.calibrated_wrist_rot, self.calibrated_finger_pose = self._compute_init_frame(
                    init_pose_buffer[:, :3], init_pose_buffer[:, 3:7],
                    init_finger_pose_buffer)

                self.init_pose_buffer.clear()
                self.init_fingier_pose_buffer.clear()

                self.last_wrist_ee_pose = None
                self.last_wrist_ee_quat = None
                intructions = "You are ready for teleoperation. Please start the teleoperation."
                if self.use_motion_filter:
                    self.wrist_pos_filter.reset()
                    self.wrist_rot_filter.reset()

            self.last_wrist_ee_pose = cur_rest_ee_pose.clone()
            self.last_wrist_ee_quat = cur_rest_ee_quat.clone()
        return intructions

    def _compute_current_hand_action(self, wrist_ee_pos, wrist_ee_rot,
                                     hand_joint_angles):

        # Debugging point to inspect wrist_ee_pos, wrist_ee_rot, hand_joint_angles
        target_ee_pose = wrist_ee_pos.clone() + self.init_ee_pose[:, :3]
        target_ee_quat = math_utils.quat_mul(wrist_ee_rot.unsqueeze(0),
                                             self.init_ee_pose[:, 3:7])
        final_actions = torch.cat(
            [target_ee_pose, target_ee_quat, hand_joint_angles], dim=-1)
        return final_actions
