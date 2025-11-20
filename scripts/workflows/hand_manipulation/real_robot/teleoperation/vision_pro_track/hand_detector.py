from scripts.workflows.hand_manipulation.real_robot.teleoperation.vision_pro_track.hand_retarget_wrapper import HandRetargetWrapper
from scripts.workflows.hand_manipulation.real_robot.teleoperation.vision_pro_track.arm_retarget_wrapper import ArmRetargetWrapper
import numpy as np

import yaml

from scripts.workflows.hand_manipulation.real_robot.teleoperation.utils.visionpro_utils import *

from pytransform3d import rotations

import time


class HandDetector(ArmRetargetWrapper):

    def __init__(self, args_cli):
        self.args_cli = args_cli

        try:
            with open(
                    f"source/config/task/hand_env/teleoperation/bunny/teleop_user_info/{self.args_cli.teleop_user}.yml",
                    'r') as file:
                self.teleop_config = yaml.safe_load(file)

        except:
            with open(
                    f"source/config/task/hand_env/teleoperation/bunny/teleop_user_info/Entong.yml",
                    'r') as file:
                self.teleop_config = yaml.safe_load(file)
        super().__init__(args_cli, self.teleop_config)
        self.hand_wrapper = HandRetargetWrapper(
            args_cli=self.args_cli,
            teleop_config=self.teleop_config,
        )

        self.initialized = False
        self.reset_mode = True

    def filter_commands(self, ):
        command = []

        if self.latest_command == {}:
            return command

        if "play" in self.latest_command:
            self.play_mode = True
            self.pause_mode = False
            self.reset_mode = False
            self.teleop_server.send_command("Begin teleoperation")
            command.append("play")
        if "pause" in self.latest_command:
            self.pause_mode = True
            self.play_mode = False
            self.reset_mode = False
            self.teleop_server.send_command("Pause teleoperation")
            command.append("pause")
        if "reset" in self.latest_command:

            self.reset_modes()
            self.teleop_server.send_command("Reset teleoperation")
            command.append("reset")
        if "replay" in self.latest_command:
            self.replay_mode = True
            self.reset_mode = False
            self.pause_mode = True
            self.teleop_server.send_command("Replay teleoperation")
            command.append("replay")
        if "save" in self.latest_command:
            self.save_mode = True
            self.pause_mode = True
            self.reset_mode = True
            self.teleop_server.send_command("Save teleoperation")
            command.append("save")
        if "remove" in self.latest_command:
            self.remove_mode = True
            self.pause_mode = True
            self.reset_mode = True
            self.teleop_server.send_command("Remove teleoperation")
            command.append("remove")
        time.sleep(0.2)  # Allow time for the command to be processed

        return command

    def on_hand_detection(self, latest_data):

        if latest_data == {}:
            self.teleop_server.send_command("No data streaming")
            return

        left_global_pose = latest_data["left_wrist"]
        right_global_pose = latest_data["right_wrist"]

        left_pos, left_quat = left_global_pose[:3,
                                               3], rotations.quaternion_from_matrix(
                                                   left_global_pose[:3, :3])
        right_pos, right_quat = right_global_pose[:3,
                                                  3], rotations.quaternion_from_matrix(
                                                      right_global_pose[:3, :3]
                                                  )

        left_hand_joints = np.reshape(latest_data["left_fingers"], [21, 3])
        right_hand_joints = np.reshape(latest_data["right_fingers"], [21, 3])

        if not self.ready_for_teleoperation and self.play_mode:

            self._init_calibration(
                np.concatenate([left_pos, left_quat]),
                left_hand_joints,
                np.concatenate([right_pos, right_quat]),
                right_hand_joints,
            )
            self.latest_info = {}
            return []
        # If the teleoperation is ready, we need to update the wrist pose
        elif self.ready_for_teleoperation and self.play_mode:
            if self.first_initialized:
                self.first_initialized = False
                self.teleop_server.send_command("Teleoperating in progress")

            if self.add_left_hand:
                left_robot_pos, left_robot_quat, delta_left_pose, delta_left_quat = self._compute_pose_in_init_frame(
                    left_pos, left_quat, hand_side="left")
                left_instruction, left_good_pose = self.evaluate_pose(
                    left_robot_pos, hand_side="left")

                delta_left_pose = np.concatenate(
                    [delta_left_pose, delta_left_quat])

            else:
                left_instruction = ""
                left_good_pose = True
            if self.add_right_hand:
                right_robot_pos, right_robot_quat, delta_right_pose, delta_right_quat = self._compute_pose_in_init_frame(
                    right_pos, right_quat, hand_side="right")
                right_instruction, right_good_pose = self.evaluate_pose(
                    right_robot_pos, hand_side="right")

                delta_right_pose = np.concatenate(
                    [delta_right_pose, delta_right_quat])

            else:
                right_instruction = ""
                right_good_pose = True

            if not right_good_pose or not left_good_pose:
                instruction = left_instruction + right_instruction
                self.teleop_server.send_command(instruction)
                self.latest_info = {}
                return []

            if self.add_left_hand:
                left_finger_pose = self.hand_wrapper.retarget_hand(
                    latest_data["left_fingers"], "left")
                left_hand_pose = np.concatenate(
                    [left_robot_pos, left_robot_quat, left_finger_pose])

            if self.add_right_hand:
                right_finger_pose = self.hand_wrapper.retarget_hand(
                    latest_data["right_fingers"], "right")
                right_hand_pose = np.concatenate(
                    [right_robot_pos, right_robot_quat, right_finger_pose])

            self.latest_info = {
                "left_hand_pose":
                left_hand_pose if self.add_left_hand else None,
                "delta_left_hand_pose":
                delta_left_pose if self.add_left_hand else None,
                "right_hand_pose":
                right_hand_pose if self.add_right_hand else None,
                "delta_right_hand_pose":
                delta_right_pose if self.add_right_hand else None,
            }

    def detect_hands(self):
        self.latest_command, latest_data = self.teleop_server.get_latest_command(
        )

        commands = self.filter_commands()

        if self.pause_mode or self.reset_mode or self.replay_mode:
            self.latest_info = {}

            return {"commands": commands}

        self.on_hand_detection(latest_data)
        return {"commands": commands}

    def process_detections(self, detections):
        # Process the raw detections and return structured data
        pass
