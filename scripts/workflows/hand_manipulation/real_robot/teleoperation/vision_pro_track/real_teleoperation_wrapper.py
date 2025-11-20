from scripts.workflows.hand_manipulation.real_robot.teleoperation.vision_pro_track.hand_detector import HandDetector
import torch
import numpy as np

# from scripts.workflows.hand_manipulation.real_robot.teleoperation.utils.visionpro_utils import IKPlanner

import time


class RealTeleoperationWrapper(HandDetector):

    def __init__(self, args_cli):
        self.args_cli = args_cli

        super().__init__(args_cli)

        from scripts.workflows.hand_manipulation.env.teleop_env.motion_planner_env import ArmMotionPlannerEnv

        arm_motion_env = ArmMotionPlannerEnv(None, args_cli, None)

        init_ee_pose = torch.as_tensor(np.array(self.init_ee_pose),
                                       device="cuda",
                                       dtype=torch.float32).unsqueeze(0)

        self.init_arm_qpos = arm_motion_env.ik_plan_motion(
            init_ee_pose).reshape(-1).cpu().numpy()
        self.last_command = {"commands": None}
        self.last_command_count = 0

    def run(self):

        command_dict = self.detect_hands()
        self.latest_robot_info = self.latest_info | {
            "init_arm_qpos": self.init_arm_qpos
        }

        if self.args_cli.send_command_to_robot:
            if (len(command_dict["commands"]) > 0 or self.last_command_count
                    > 0) and self.last_command_count < 20:
                if self.last_command_count == 0:
                    print(command_dict["commands"])
                    self.last_command = command_dict
                self.latest_robot_info = self.latest_robot_info | self.last_command
                self.last_command_count += 1

            else:
                self.latest_robot_info = self.latest_robot_info | command_dict

            if self.last_command_count >= 20:
                self.last_command_count = 0
            # print(self.latest_robot_info["commands"])

            self.teleop_server.send_teleop_cmd(self.latest_robot_info)
