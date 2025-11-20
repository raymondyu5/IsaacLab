from scripts.workflows.hand_manipulation.env.teleop_env.motion_planner_env import ArmMotionPlannerEnv
from scripts.workflows.hand_manipulation.real_robot.teleoperation.utils.pinocchio_motion_control import PinocchioMotionControl
from scripts.workflows.hand_manipulation.real_robot.teleoperation.teleop_client import TeleopClient
import numpy as np

import copy

import torch
import isaaclab.utils.math as math_utils
from scipy.spatial.transform import Rotation as R


class MeshcatRobotController(TeleopClient):

    def __init__(self,
                 port,
                 host,
                 action_space,
                 add_right_hand=True,
                 add_left_hand=False):

        # self.arm_motion_env = ArmMotionPlannerEnv()
        self.arm_motion_pinco = PinocchioMotionControl(
            "scripts/workflows/hand_manipulation/real_robot/teleoperation/utils/franka_leap.yaml"
        )
        self.action_space = action_space

        self.last_action = np.zeros(23)
        self.last_joint_pos = None
        self.action_space = action_space
        self.add_right_hand = add_right_hand
        self.add_left_hand = add_left_hand
        self.action_buffer = []
        self.quat_offset = torch.as_tensor(
            [0.707, 0.707, 0.0, 0.0],
            dtype=torch.float32).to("cuda").unsqueeze(0)
        super().__init__(port, host)

    def get_teleop_data(self):

        teleop_data = self.get_teleop_cmd()

        if "right_hand_pose" not in teleop_data and "left_hand_pose" not in teleop_data:  # no hands detected keep the last action

            command = teleop_data["commands"]

            return [self.last_action], command

        if self.arm_motion_pinco.qpos is None:
            init_arm_qpos = copy.deepcopy(
                np.array(teleop_data["init_arm_qpos"]).reshape(7))

            self.arm_motion_pinco.qpos = np.concatenate(
                [init_arm_qpos, np.zeros(16)])
        command = teleop_data["commands"]

        if command == "reset":
            init_arm_qpos = copy.deepcopy(
                np.array(teleop_data["init_arm_qpos"]).reshape(7))
            self.last_action = np.concatenate([init_arm_qpos, np.zeros(16)])
            self.action_buffer.clear()

            return [np.concatenate([init_arm_qpos, np.zeros(16)])], command
        if command == "pause":

            return [self.last_action], command

        if command == "replay":
            if len(self.action_buffer) > 0:
                return self.action_buffer
            return [self.last_action], command

        if self.add_left_hand:
            action = teleop_data["left_hand_pose"]

        if self.add_right_hand:
            action = teleop_data["right_hand_pose"]

        arm_qpos = self.arm_motion_pinco.step(action[:3], action[3:7])
        robot_action = np.concatenate([arm_qpos[:7], action[7:]])
        self.action_buffer.append(robot_action)
        self.last_action = robot_action
        return [robot_action], command
