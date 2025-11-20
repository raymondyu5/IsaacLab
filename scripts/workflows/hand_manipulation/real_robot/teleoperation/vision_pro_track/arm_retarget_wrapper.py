import numpy as np

import yaml
import time
from scripts.workflows.hand_manipulation.real_robot.teleoperation.utils.visionpro_utils import *
from scripts.workflows.hand_manipulation.real_robot.teleoperation.vision_pro_track.real_teleoperation_server import TeleopServer

import copy


class ArmRetargetWrapper:

    def __init__(self, args_cli, teleop_config):
        self.args_cli = args_cli

        self.teleop_server = TeleopServer(
            host_ip=args_cli.host_ip,
            port=args_cli.port,
            avp_address=args_cli.avp_address,
            send_command_to_robot=args_cli.send_command_to_robot)
        self.teleop_server.start()
        # while True:
        #     cmd, pose = self.teleop_server.get_latest_command()
        #     # if cmd:
        #     print("Latest command:", pose)
        #     time.sleep(0.1)
        self.teleop_config = teleop_config

        self.reset_modes()

        # Initial calibration

        self.add_left_hand = args_cli.add_left_hand
        self.add_right_hand = args_cli.add_right_hand

        self.pose_factor = self.teleop_config.get("pose_factor", 1.0)
        low_pass_smoothing_wrist = self.teleop_config.get(
            "low_pass_smoothing_wrist", 0.1)
        self.use_motion_filter = self.teleop_config.get(
            "use_motion_filter", False)  # for debugging
        self.init_ee_pose = self.teleop_config.get(
            "init_ee_pose",
            np.array([0.300, -0.000, 0.400, 0.0, 9.2460e-01, -3.8094e-01,
                      0.0]))
        if self.use_motion_filter:
            self.wrist_pos_filter = LPFilter(alpha=low_pass_smoothing_wrist)
            self.wrist_rot_filter = LPRotationFilter(
                alpha=low_pass_smoothing_wrist)

    def reset_modes(self):
        self.replay_mode = False
        self.reset_mode = True
        self.pause_mode = True
        self.play_mode = False
        self.save_mode = False
        self.remove_mode = False
        self.first_initialized = True
        self.begin_initialization = True

        self.last_wrist_ee_pose = None
        self.last_wrist_ee_quat = None
        self.ready_for_teleoperation = False

        for side in ["left", "right"]:
            self._set_side_attrs(side)

    def _set_side_attrs(self, side):
        setattr(self, f"{side}_ready_for_teleoperation", False)
        setattr(self, f"{side}_ready_for_teleoperation_counter", 0)
        setattr(self, f"{side}_calibrated_wrist_pos", None)
        setattr(self, f"init_{side}_hand_pose_buffer", [])
        setattr(self, f"last_{side}_wrist_ee_pose", None)
        setattr(self, f"last_{side}_wrist_ee_quat", None)

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
                            init_frame_pos: np.ndarray,
                            init_frame_quat: np.ndarray,
                            align_gravity_dir=True) -> tuple:
        num_data = init_frame_pos.shape[0]
        weight = ((np.arange(num_data) + 1) / np.sum(np.arange(num_data) + 1))

        if align_gravity_dir:
            calibrated_wrist_rot = project_average_rotation(init_frame_quat)
        else:
            calibrated_wrist_rot = rotations.matrix_from_quaternion(
                init_frame_quat[-1])

        calibrated_wrist_pos = np.sum(weight[:, None] * init_frame_pos, axis=0)

        return calibrated_wrist_pos.reshape(3), calibrated_wrist_rot

    def evaluate_init_pose(self,
                           ee_pose: np.ndarray,
                           joint_angles: np.ndarray,
                           hand_side: str,
                           dist_threshold=0.05,
                           angle_threshold=20):
        instructions = "Please place your hand in the initial position, Caliberating the hand pose."

        # Get references to internal state
        ready_for_teleoperation_counter = getattr(
            self, f"{hand_side}_ready_for_teleoperation_counter")
        init_pose_buffer = getattr(self, f"init_{hand_side}_hand_pose_buffer")
        last_wrist_ee_pose = getattr(self, f"last_{hand_side}_wrist_ee_pose")
        ready_for_teleoperation = getattr(
            self, f"{hand_side}_ready_for_teleoperation")

        cur_rest_ee_pose = ee_pose[:3]
        cur_rest_ee_quat = ee_pose[3:7]
        calibrated_wrist_pos = None
        calibrated_wrist_rot = None

        if last_wrist_ee_pose is None:
            # First time seen — store and return
            setattr(self, f"last_{hand_side}_wrist_ee_pose", cur_rest_ee_pose)
            setattr(self, f"last_{hand_side}_wrist_ee_quat", cur_rest_ee_quat)
            return instructions

        # Movement check
        moving_dist = np.linalg.norm(cur_rest_ee_pose - last_wrist_ee_pose)
        flat_threshold = (0.01, np.deg2rad(angle_threshold))
        finger_angles = self._compute_hand_joint_angles(joint_angles)
        hand_flat_spread = flat_threshold[0] < np.mean(
            finger_angles) < flat_threshold[1]
        not_moving = moving_dist < dist_threshold

        if not_moving and hand_flat_spread:
            ready_for_teleoperation_counter += 1
            init_pose_buffer.append(copy.deepcopy(ee_pose))
        else:
            ready_for_teleoperation_counter = 0
            init_pose_buffer.clear()

        if not_moving and hand_flat_spread and ready_for_teleoperation_counter > 100:
            print(f"[INFO]: Ready for teleoperation")
            setattr(self, f"{hand_side}_ready_for_teleoperation", True)
            ready_for_teleoperation_counter = 0

            init_pose_buffer_np = np.stack(init_pose_buffer)
            calibrated_wrist_pos, calibrated_wrist_rot = self._compute_init_frame(
                init_pose_buffer_np[:, :3], init_pose_buffer_np[:, 3:7])

            init_pose_buffer.clear()

            # Reset internal buffers and set calibrated results
            setattr(self, f"{hand_side}_calibrated_wrist_pos",
                    calibrated_wrist_pos)
            setattr(self, f"{hand_side}_calibrated_wrist_rot",
                    calibrated_wrist_rot)
            setattr(self, f"last_{hand_side}_wrist_ee_pose", None)
            setattr(self, f"last_{hand_side}_wrist_ee_quat", None)

            instructions = "You are ready for teleoperation. Please start the teleoperation."

            if self.use_motion_filter:
                self.wrist_pos_filter.reset()
                self.wrist_rot_filter.reset()

        # Save updated state
        setattr(self, f"last_{hand_side}_wrist_ee_pose", cur_rest_ee_pose)
        setattr(self, f"last_{hand_side}_wrist_ee_quat", cur_rest_ee_quat)
        setattr(self, f"init_{hand_side}_hand_pose_buffer", init_pose_buffer)
        setattr(self, f"{hand_side}_ready_for_teleoperation_counter",
                ready_for_teleoperation_counter)

        return instructions

    def _init_calibration(
        self,
        left_hand_wrist_pose: np.ndarray,
        left_hand_joints: np.ndarray,
        right_hand_wrist_pose: np.ndarray,
        right_hand_joints: np.ndarray,
    ):

        # DEBUG: Check the initial calibration poses
        if not self.ready_for_teleoperation:
            if self.add_left_hand:
                left_intructions = self.evaluate_init_pose(
                    left_hand_wrist_pose,
                    left_hand_joints,
                    hand_side="left",
                )
            else:
                left_intructions = ""
                self.left_ready_for_teleoperation = True

            if self.add_right_hand:
                right_intructions = self.evaluate_init_pose(
                    right_hand_wrist_pose,
                    right_hand_joints,
                    hand_side="right",
                )
            else:
                right_intructions = ""
                self.right_ready_for_teleoperation = True
            self.ready_for_teleoperation = self.left_ready_for_teleoperation and self.right_ready_for_teleoperation
            intruction = left_intructions + right_intructions

            # if self.begin_initialization:
            self.teleop_server.send_command(intruction)
            self.begin_initialization = False
            time.sleep(0.02)

    def _compute_pose_in_init_frame(
        self,
        pos: np.ndarray,
        quat: np.ndarray,
        hand_side: str,
    ):

        calibrated_init_rot = getattr(self,
                                      f"{hand_side}_calibrated_wrist_rot")
        calibrated_init_pos = getattr(self,
                                      f"{hand_side}_calibrated_wrist_pos")

        operator_rot = rotations.matrix_from_quaternion(quat)
        local_operator_rot = operator_rot @ calibrated_init_rot.T
        robot_pos = pos - calibrated_init_pos + self.init_ee_pose[:3]
        robot_rot = local_operator_rot @ rotations.matrix_from_quaternion(
            self.init_ee_pose[3:7])

        robot_quat = rotations.quaternion_from_matrix(robot_rot)
        if self.use_motion_filter:

            robot_pos = self.wrist_pos_filter.next(robot_pos)
            robot_quat = self.wrist_rot_filter.next(robot_quat)

        local_operator_quat = rotations.quaternion_from_matrix(
            local_operator_rot)

        return robot_pos, robot_quat, pos - calibrated_init_pos, local_operator_quat

    def evaluate_pose(self,
                      ee_pose,
                      hand_side: str,
                      bbox=[0.1, -0.3, 0.0, 0.85, 0.3, 0.6]):

        if ee_pose[0] > bbox[3] or ee_pose[0] < bbox[0]:

            return f"Stop Teleoperation, Your {hand_side} hand x location is {ee_pose[0]:.2f} . Please reset the teleoperation.", False
        if ee_pose[1] > bbox[4] or ee_pose[1] < bbox[1]:

            return f"Stop Teleoperation, Your {hand_side} hand y location is {ee_pose[1]:.2f} . Please reset the teleoperation.", False
        if ee_pose[2] > bbox[-1]:

            return f"Stop Teleoperation, Your {hand_side} hand z location is {ee_pose[2]:.2f} . Please reset the teleoperation.", False
        dist_to_init = np.linalg.norm(ee_pose[:3] - self.init_ee_pose[:3], )
        if dist_to_init > 0.4:

            return f"Stop Teleoperation, Your {hand_side} hand hand is too far from the initial position. Please reset the teleoperation.", False
        return None, True
