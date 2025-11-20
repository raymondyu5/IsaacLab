import isaaclab.utils.math as math_utils

import torch

import numpy as np


class ActionManagerWrapper:

    def __init__(self, env, args_cli, env_cfg, teleop_config,
                 visionpro_client):
        self.env = env
        self.args_cli = args_cli
        self.env_cfg = env_cfg
        self.teleop_config = teleop_config
        self.visionpro_client = visionpro_client

        self.add_left_hand = args_cli.add_left_hand
        self.add_right_hand = args_cli.add_right_hand
        self.device = args_cli.device

        if self.args_cli.calibration:
            self.step_action = self.step_rel_action

        else:
            self.step_action = self.step_abs_action

        self.init_reset_teleoperation()
        self.init_filter()

    def init_reset_teleoperation(self):

        ######### reset action
        from scripts.workflows.hand_manipulation.env.teleop_env.motion_planner_env import ArmMotionPlannerEnv
        self.arm_motion_env = ArmMotionPlannerEnv(
            self.env,
            self.args_cli,
            self.env_cfg,
        )
        self.num_hand_joint = self.env_cfg["params"]["num_hand_joints"]

        init_ee_pose = torch.as_tensor(
            self.env_cfg["params"]["init_ee_pose"]).to(
                self.device).unsqueeze(0)

        init_pose = torch.cat([
            init_ee_pose,
            torch.zeros(1, self.num_hand_joint).to(self.device)
        ],
                              dim=1).repeat_interleave(self.env.num_envs,
                                                       dim=0)
        init_arm_qpos = self.arm_motion_env.ik_plan_motion(
            init_ee_pose).repeat_interleave(self.env.num_envs, dim=0)

        self.init_actions = []
        if self.add_left_hand:

            self.init_actions.append(init_pose)

        if self.add_right_hand:

            self.init_actions.append(init_pose)

        self.init_actions = torch.cat(self.init_actions, dim=1)

        if self.args_cli.add_left_hand:
            self.leftpalm2wrist, self.leftwrist2palm = self.init_delta_transform(
                "left")
            self.env.scene["left_hand"]._data.reset_joint_pos = torch.cat(
                [
                    init_arm_qpos.reshape(-1, 7),
                    torch.zeros(1, self.num_hand_joint).to(self.device)
                ],
                dim=1).repeat_interleave(self.env.num_envs, dim=0)

        if self.args_cli.add_right_hand:
            self.rightpalm2wrist, self.rightwrist2palm = self.init_delta_transform(
                "right")

            self.env.scene["right_hand"]._data.reset_joint_pos = torch.cat(
                [
                    init_arm_qpos.reshape(-1, 7),
                    torch.zeros(1, self.num_hand_joint).to(self.device)
                ],
                dim=1).repeat_interleave(self.env.num_envs, dim=0)

        ######### teleop offset
        self.left_hand_quat_offset = torch.tensor([
            .0000, 0.7070, -0.7070, 0.0000
        ]).to(device=self.env.device).unsqueeze(0).repeat_interleave(
            self.env.num_envs, dim=0)

        self.right_hand_quat_offset = torch.tensor([
            0.0000, 0.0000, 0.7071, 0.7071
        ]).to(device=self.env.device).unsqueeze(0).repeat_interleave(
            self.env.num_envs, dim=0)
        self.right_hand_quat_offset = math_utils.quat_mul(
            self.right_hand_quat_offset,
            torch.tensor([[0.9238795, 0, 0,
                           -0.3826834]]).to(device=self.env.device))

        self.left_hand_pos_offset = torch.tensor(
            [0.0, 0.0,
             0.0]).to(device=self.env.device).unsqueeze(0).repeat_interleave(
                 self.env.num_envs, dim=0)
        self.right_hand_pos_offset = torch.tensor(
            [0.0, 0.0,
             0.0]).to(device=self.env.device).unsqueeze(0).repeat_interleave(
                 self.env.num_envs, dim=0)

    def init_filter(self):

        ######### action filter
        if self.args_cli.calibration:

            self.low_pass_smoothing_wrist = self.teleop_config.get(
                "low_pass_smoothing_wrist", 0.1)
            self.low_pass_smoothing_finger = self.teleop_config.get(
                "low_pass_smoothing_finger", 0.5)
            wrist_pos_filter = math_utils.LPFilter(
                alpha=self.low_pass_smoothing_wrist, )

            wrist_rot_filter = math_utils.LPRotationFilter(
                alpha=self.low_pass_smoothing_wrist, )
            finger_filter = math_utils.LPFilter(
                alpha=self.low_pass_smoothing_finger)

            if self.add_left_hand:
                self.left_wrist_pos_filter = wrist_pos_filter
                self.left_wrist_rot_filter = wrist_rot_filter
                self.left_finger_filter = finger_filter
                self.init_left_hand_pose_buffer = []

            if self.add_right_hand:
                self.right_wrist_pos_filter = wrist_pos_filter
                self.right_wrist_rot_filter = wrist_rot_filter
                self.right_finger_filter = finger_filter
                self.init_right_hand_pose_buffer = []
            self.reset()

    def apply_offset(self, teleop_data, hand_side):
        palm2wrist = getattr(self, f"{hand_side}palm2wrist")
        hand_quat_offset = getattr(self, f"{hand_side}_hand_quat_offset")
        hand_pos_offset = getattr(self, f"{hand_side}_hand_pos_offset")

        hand_pose = torch.zeros((self.env.num_envs, 7), device=self.device)
        hand_pose[..., 3:7] = math_utils.quat_mul(
            torch.as_tensor(teleop_data[3:7]).to(
                self.device).unsqueeze(0).repeat_interleave(self.env.num_envs,
                                                            dim=0),
            hand_quat_offset)

        hand_pose[..., :3] = torch.as_tensor(teleop_data[:3]).to(
            self.device).unsqueeze(0).repeat_interleave(
                self.env.num_envs, dim=0) + hand_pos_offset

        wrist_ee_pose = torch.cat(math_utils.combine_frame_transforms(
            hand_pose[:, :3], hand_pose[:, 3:7], palm2wrist[:, :3],
            palm2wrist[:, 3:7]),
                                  dim=1)

        wrist_ee_pose[..., :3], wrist_ee_pose[
            ..., 3:7] = math_utils.combine_frame_transforms(
                wrist_ee_pose[..., :3], wrist_ee_pose[..., 3:7],
                torch.tensor([[0.0000, 0.0000, -0.1070]], device=self.device),
                torch.tensor([[1., 0., 0., 0.]], device=self.device))

        return wrist_ee_pose

    def init_delta_transform(self, hand_side):
        palm_state = self.env.scene[
            f"{hand_side}_palm_lower"]._data.root_state_w[:, :7]
        wrist_state = self.env.scene[
            f"{hand_side}_panda_link7"]._data.root_state_w[:, :7]
        palm2wrist = torch.cat(math_utils.subtract_frame_transforms(
            palm_state[:, :3], palm_state[:, 3:7], wrist_state[:, :3],
            wrist_state[:, 3:7]),
                               dim=1)
        wrist2palm = torch.cat(math_utils.subtract_frame_transforms(
            wrist_state[:, :3], wrist_state[:, 3:7], palm_state[:, :3],
            palm_state[:, 3:7]),
                               dim=1)

        return palm2wrist, wrist2palm

    def reset(self):
        self.ready_for_teleop = True
        if self.args_cli.calibration:
            if self.add_left_hand:
                self.left_wrist_pos_filter.reset()
                self.left_wrist_rot_filter.reset()
                self.left_finger_filter.reset()
                self.left_finger_filter(
                    torch.zeros((self.env.num_envs, 16)).to(self.device))
                self.last_left_wrist_ee_pose = None
            if self.add_right_hand:
                self.right_wrist_pos_filter.reset()
                self.right_wrist_rot_filter.reset()
                self.right_finger_filter.reset()
                self.right_finger_filter(
                    torch.zeros((self.env.num_envs, 16)).to(self.device))
                self.last_right_wrist_ee_pose = None
            self.ready_for_teleop = False
            self.calibration_steps = 0

        if self.add_left_hand:
            self.prev_left_wrist_pose = self.env.scene[
                "left_panda_link7"]._data.root_state_w[:, :7]
        if self.add_right_hand:
            self.prev_right_wrist_pose = self.env.scene[
                "right_panda_link7"]._data.root_state_w[:, :7]

    @staticmethod
    def _compute_hand_joint_angles(joints: torch.Tensor) -> torch.Tensor:
        """
        Compute the angle between fingertip vectors and palm bone vectors.

        Args:
            joints: torch.Tensor of shape (N, 3), where N ≥ 21 (hand joints in 3D).

        Returns:
            torch.Tensor of shape (5,), angles in radians.
        """
        device = joints.device
        tip_index = torch.tensor([5, 10, 15, 20, 25], device=device)
        palm_bone_index = torch.tensor([2, 7, 14, 19, 24], device=device)

        root = joints[0:1, :]  # (1, 3)
        tips = joints[tip_index]  # (5, 3)
        palm_bone = joints[palm_bone_index]  # (5, 3)

        tip_vec = tips - root  # (5, 3)
        tip_vec = tip_vec / torch.norm(tip_vec, dim=1, keepdim=True)

        palm_vec = palm_bone - root  # (5, 3)
        palm_vec = palm_vec / torch.norm(palm_vec, dim=1, keepdim=True)

        dot = torch.sum(tip_vec * palm_vec, dim=1)  # (5,)
        dot = torch.clamp(dot, -1.0, 1.0)  # numerical stability
        angle = torch.acos(dot)

        return angle

    def project_average_rotation(self,
                                 quat_list: torch.Tensor) -> torch.Tensor:
        """
            Args:
                quat_list: (N, 4) torch tensor of quaternions (w, x, y, z) or (x, y, z, w) 
                        depending on your rotations.matrix_from_quaternion convention.

            Returns:
                final_mat: (3, 3) torch tensor, averaged calibrated rotation matrix.
            """
        device = quat_list.device
        dtype = quat_list.dtype

        gravity_dir = torch.tensor([0.0, 0.0, -1.0],
                                   device=device,
                                   dtype=dtype)

        # last quaternion → rotation matrix
        last_quat = quat_list[-1, :]
        last_mat = math_utils.matrix_from_quat(last_quat)  # (3, 3)

        # gravity alignment
        gravity_quantity = gravity_dir @ last_mat  # (3,)
        max_gravity_axis = torch.argmax(torch.abs(gravity_quantity)).item()
        same_direction = gravity_quantity[max_gravity_axis] > 0

        next_axis = (max_gravity_axis + 1) % 3
        next_next_axis = (max_gravity_axis + 2) % 3

        # collect projected angles
        angles = []
        for i in range(quat_list.shape[0]):
            mat = math_utils.matrix_from_quat(quat_list[i])  # (3, 3)
            next_dir = mat[:3, next_axis]  # vector
            next_dir = next_dir.clone()
            next_dir[2] = 0.0  # project to XY plane

            # spherical_from_cartesian → azimuth angle = atan2(y, x)
            next_dir_angle = torch.atan2(next_dir[1], next_dir[0])
            angles.append(next_dir_angle)

        angle = torch.stack(angles).mean()

        # construct final rotation matrix
        final_mat = torch.zeros((3, 3), device=device, dtype=dtype)
        final_mat[:, max_gravity_axis] = gravity_dir * (1.0 if same_direction
                                                        else -1.0)
        final_mat[:, next_axis] = torch.tensor(
            [torch.cos(angle), torch.sin(angle), 0.0],
            device=device,
            dtype=dtype)
        final_mat[:,
                  next_next_axis] = torch.cross(final_mat[:, max_gravity_axis],
                                                final_mat[:, next_axis])

        return math_utils.quat_from_matrix(final_mat)

    def _compute_init_frame(self,
                            init_frame_pos: torch.Tensor,
                            init_frame_quat: torch.Tensor,
                            align_gravity_dir: bool = True) -> tuple:
        """
        Args:
            init_frame_pos: (N, 3) torch tensor of positions
            init_frame_quat: (N, 4) torch tensor of quaternions
            align_gravity_dir: whether to project average rotation

        Returns:
            calibrated_wrist_pos: (3,) torch tensor
            calibrated_wrist_rot: (3, 3) torch tensor (rotation matrix)
        """
        device = init_frame_pos.device
        num_data = init_frame_pos.shape[0]

        # weights: (N,)
        weight = (
            torch.arange(num_data, device=device, dtype=init_frame_pos.dtype) +
            1)
        weight = weight / weight.sum()

        # rotation calibration
        if align_gravity_dir:
            # assumes you have a torch implementation of project_average_rotation
            calibrated_wrist_rot = self.project_average_rotation(
                init_frame_quat)
        else:
            # assumes you have a torch implementation of quaternion -> rotation matrix
            calibrated_wrist_rot = init_frame_quat[-1]

        # weighted average position
        calibrated_wrist_pos = torch.sum(weight[:, None] * init_frame_pos,
                                         dim=0)

        return calibrated_wrist_pos.reshape(3), calibrated_wrist_rot

    def evaluate_init_pose(self,
                           ee_pose: torch.Tensor,
                           hand_side: str,
                           dist_threshold: float = 0.05,
                           angle_threshold: float = 20.0,
                           raw_data: dict = None):

        if self.calibration_steps == 0:
            self.visionpro_client.send_command(
                command="Caliberating init pose for {}".format(hand_side))

        init_pose_buffer = getattr(self, f"init_{hand_side}_hand_pose_buffer")
        last_wrist_ee_pose = getattr(self, f"last_{hand_side}_wrist_ee_pose")

        cur_rest_ee_pose = ee_pose[..., :3]
        cur_rest_ee_quat = ee_pose[..., 3:7]
        calibrated_wrist_pos = None
        calibrated_wrist_rot = None

        if last_wrist_ee_pose is None:
            # First time seen — store and return
            setattr(self, f"last_{hand_side}_wrist_ee_pose", cur_rest_ee_pose)
            setattr(self, f"last_{hand_side}_wrist_ee_quat", cur_rest_ee_quat)

        # Movement check

        last_wrist_ee_pose = getattr(self, f"last_{hand_side}_wrist_ee_pose")
        moving_dist = torch.norm(cur_rest_ee_pose - last_wrist_ee_pose)
        flat_threshold = (0.01, torch.deg2rad(torch.tensor(angle_threshold)))
        values = list(raw_data.values())
        flattened = torch.from_numpy(np.array(values))
        finger_angles = self._compute_hand_joint_angles(flattened)
        mean_finger = torch.mean(finger_angles)
        hand_flat_spread = (flat_threshold[0] < mean_finger <
                            flat_threshold[1])
        not_moving = moving_dist < dist_threshold

        if not_moving and hand_flat_spread:
            self.calibration_steps += 1
            init_pose_buffer.append(ee_pose.clone())
        else:
            self.calibration_steps += 0

            init_pose_buffer.clear()

        if not_moving and hand_flat_spread and self.calibration_steps > 25:
            print(f"[INFO]: Ready for teleoperation")
            setattr(self, f"{hand_side}_ready_for_teleoperation", True)

            if self.calibration_steps == 51:

                self.visionpro_client.send_command(
                    command="Ready for Teleoperation")

            init_pose_buffer_t = torch.stack(init_pose_buffer)
            calibrated_wrist_pos, calibrated_wrist_rot = self._compute_init_frame(
                init_pose_buffer_t[:, 0, :3], init_pose_buffer_t[:, 0, 3:7],
                self.teleop_config.get("align_gravity_dir", True))

            init_pose_buffer.clear()

            # Reset internal buffers and set calibrated results
            setattr(self, f"{hand_side}_calibrated_wrist_pos",
                    calibrated_wrist_pos.unsqueeze(0))
            setattr(self, f"{hand_side}_calibrated_wrist_rot",
                    calibrated_wrist_rot.unsqueeze(0))
            setattr(self, f"last_{hand_side}_wrist_ee_pose", None)
            setattr(self, f"last_{hand_side}_wrist_ee_quat", None)

        # Save updated state
        setattr(self, f"last_{hand_side}_wrist_ee_pose",
                cur_rest_ee_pose.clone())
        setattr(self, f"last_{hand_side}_wrist_ee_quat",
                cur_rest_ee_quat.clone())
        setattr(self, f"init_{hand_side}_hand_pose_buffer", init_pose_buffer)

    def _compute_pose_in_init_frame(
        self,
        pos: torch.Tensor,  # (3,)
        quat: torch.Tensor,  # (4,)
        hand_side: str,
    ):
        """
        Compute wrist pose relative to the calibrated init frame.

        Args:
            pos: (3,) torch tensor, current wrist position
            quat: (4,) torch tensor, current wrist quaternion
            hand_side: "right" or "left"

        Returns:
            delta_pos: (3,) torch tensor
            local_operator_quat: (4,) torch tensor
        """
        # Get calibrated wrist pose (must be torch tensors)
        calibrated_init_rot: torch.Tensor = getattr(
            self, f"{hand_side}_calibrated_wrist_rot")  # (3,3)
        calibrated_init_pos: torch.Tensor = getattr(
            self, f"{hand_side}_calibrated_wrist_pos")  # (3,)

        delta_pos = math_utils.extract_delta_pose(
            torch.cat([pos, quat], dim=1),
            torch.cat([calibrated_init_pos, calibrated_init_rot], dim=1))

        return delta_pos

    def process_hand_data(self,
                          pos: torch.Tensor,
                          quat: torch.Tensor,
                          finger_pose: torch.Tensor,
                          hand_side: str = "right") -> torch.Tensor:
        """
        Process hand data into a unified hand pose vector.

        Args:
            pos: (3,) torch tensor, hand position
            quat: (4,) torch tensor, hand quaternion (xyzw format expected downstream)
            finger_pose: (N,) torch tensor, finger joint angles
            hand_side: "right" or "left"

        Returns:
            hand_pose: (M,) torch tensor, concatenated [delta_pose (3), delta_rot (3), finger_pose (N)]
        """
        # Pose in init frame
        delta_pose = self._compute_pose_in_init_frame(pos,
                                                      quat,
                                                      hand_side=hand_side)

        target_pos, target_rot = math_utils.apply_delta_pose(
            self.init_actions[:, :3],
            self.init_actions[:, 3:7],
            delta_pose,
        )

        finger_filter = getattr(self, f"{hand_side}_finger_filter")

        filter_finger_pose = finger_filter(finger_pose)
        translate_filter = getattr(self, f"{hand_side}_wrist_pos_filter")
        rotate_filter = getattr(self, f"{hand_side}_wrist_rot_filter")
        target_pos = translate_filter(target_pos)
        target_rot = rotate_filter(target_rot)

        if self.args_cli.max_step is not None:
            # Ensure the filtered position does not exceed max_step from previous position

            cur_ee_pose = self.env.scene[
                f"{hand_side}_panda_link7"]._data.root_state_w[:, :7]

            delta_pose = math_utils.extract_delta_pose(
                torch.cat([target_pos, target_rot], dim=1), cur_ee_pose)

            delta_pose[:, :3] = torch.clamp(delta_pose[:, :3],
                                            -self.args_cli.max_step[0],
                                            self.args_cli.max_step[0])
            delta_pose[:, 3:6] = torch.clamp(delta_pose[:, 3:6],
                                             -self.args_cli.max_step[1],
                                             self.args_cli.max_step[1])
            target_pos, target_rot = math_utils.apply_delta_pose(
                cur_ee_pose[:, :3], cur_ee_pose[:, 3:7], delta_pose)

        # Full hand pose
        hand_pose = torch.cat([target_pos, target_rot, filter_finger_pose],
                              dim=-1)

        return hand_pose

    def step_rel_action(self, arm_data, hand_data, hand_side, raw_data):

        wrist_pose = self.apply_offset(arm_data, hand_side)

        if not self.ready_for_teleop:

            self.evaluate_init_pose(ee_pose=wrist_pose,
                                    hand_side=hand_side,
                                    raw_data=raw_data)

            if self.calibration_steps > 25:
                self.ready_for_teleop = True
                self.visionpro_client.send_command(
                    command="Teleoperation in progress".format(hand_side))
            return self.init_actions

        return self.process_hand_data(
            wrist_pose[:, :3],
            wrist_pose[:, 3:7],
            torch.as_tensor(hand_data).to(self.device),
            hand_side=hand_side).repeat_interleave(self.env.num_envs, dim=0)

    def step_abs_action(self, arm_data, hand_data, hand_side, raw_data):

        wrist_pose = self.apply_offset(arm_data, hand_side)
        hand_pose = torch.cat([wrist_pose, hand_data], dim=1)

        return hand_pose
