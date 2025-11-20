import torch
import isaaclab.utils.math as math_utils
from isaaclab.envs import mdp
from isaaclab.sensors.camera.utils import obtain_target_quat_from_multi_angles
from scripts.workflows.hand_manipulation.real_robot.teleoperation.ready_teleoperation import ReadyTeleoperation, project_average_rotation

import numpy as np


class RealArmRetargeter(ReadyTeleoperation):

    def __init__(
        self,
        env,
        teleop_interface,
        args_cli,
        env_cfg,
        init_hand_dir="face_down",
        teleop_config=None,
    ):

        super().__init__(env, teleop_interface, args_cli, env_cfg,
                         teleop_config, init_hand_dir)

    def extract_action_info(self,
                            actions,
                            raw_data,
                            good_pose,
                            hand_side="right"):

        if not self.ready_for_teleoperation or not good_pose:
            return {}

        action_dict = {}
        action_dict["ee_actions"] = actions.cpu().numpy()

        ee_quat_des = self.env.action_manager._terms[
            f"{hand_side}_arm_action"]._ik_controller.ee_quat_des.clone()
        ee_pos_des = self.env.action_manager._terms[
            f"{hand_side}_arm_action"]._ik_controller.ee_pos_des.clone()
        joint_pos_des = self.env.action_manager._terms[
            f"{hand_side}_arm_action"].joint_pos_des.clone()
        finger_pos_des = self.env.action_manager._terms[
            f"{hand_side}_hand_action"].processed_actions.clone()
        action_dict["ee_control_action"] = torch.cat(
            [ee_pos_des, ee_quat_des, finger_pos_des], dim=-1).cpu().numpy()
        action_dict["joint_control_action"] = torch.cat(
            [joint_pos_des, finger_pos_des], dim=-1).cpu().numpy()

        action_dict["raw_hand_data"] = raw_data

        return action_dict

    def evaluate_pose(
        self,
        actions,
    ):
        ee_pose = actions[0, :3].cpu().numpy()

        if ee_pose[0] > 0.65 or ee_pose[0] < 0.1:

            return f"Stop Teleoperation, Your x location is {ee_pose[0]} . Please reset the teleoperation.", False
        if ee_pose[1] > 0.3 or ee_pose[1] < -0.3:

            return f"Stop Teleoperation, Your y location is {ee_pose[1]} . Please reset the teleoperation.", False
        if ee_pose[2] > 0.6:

            return f"Stop Teleoperation, Your z location is {ee_pose[2]} . Please reset the teleoperation.", False
        dist_to_init = torch.norm(
            actions[0, :3] - self.init_ee_pose[0, :3].to(torch.float32),
            dim=-1,
        )
        if dist_to_init > 0.4:

            return "Stop Teleoperation, Your hand is too far from the initial position. Please reset the teleoperation."
        return None, True

    def step_active_teleoperation(self, teleop_data, raw_data):

        actions = []
        intructions = None

        right_wrist_pose = self.apply_offset(teleop_data[0][1])

        right_hand_pose = torch.cat([
            right_wrist_pose,
            torch.as_tensor(teleop_data[0][-1][-self.num_hand_joint:]).to(
                self.device).unsqueeze(0)
        ],
                                    dim=1)

        actions.append(right_hand_pose)

        actions = torch.cat(actions, dim=1)
        good_pose = False

        if not self.ready_for_teleoperation:

            obs, rewards, terminated, time_outs, extras = self.env.step(
                self.init_actions)

            intructions = self.evaluate_init_pose(actions[
                0,
            ], np.array(teleop_data[0][-2]))
        else:

            wrist_ee_pos, wrist_ee_rot, local_operator_finger_pose = self._compute_pose_in_init_frame(
                right_hand_pose[0, :3].to(torch.float32),
                right_hand_pose[0, 3:7].to(torch.float32),
                actions[0, 7:].to(torch.float32))

            step_action = self._compute_current_hand_action(
                wrist_ee_pos * self.pose_factor,
                wrist_ee_rot,
                actions[:, 7:],
            )

            obs, rewards, terminated, time_outs, extras = self.env.step(
                step_action)
            intructions, good_pose = self.evaluate_pose(step_action)

        obs["policy"]["raw_teleop_data"] = raw_data

        obs["policy"]["dexretargeting_human_data"] = teleop_data[0][-2]
        actions_dict = self.extract_action_info(actions, teleop_data[0][-2],
                                                good_pose)
        return obs, actions_dict, False, intructions

    def step_reset_teleoperation(self, ):

        self.ready_for_teleoperation = False

        self.env.reset()

        for i in range(10):
            self.env.step(self.init_actions)
        reset = True

        print("Resetting environment...", self.num_frame)
        self.num_frame = 0
        self.teleop_counter = 0
        intructions = "Reset Finished Please place your hand in the initial position"
        return None, {}, reset, intructions

    def step_teleoperation(self, teleoperation_active,
                           reset_stopped_teleoperation):
        intructions = None

        teleop_data, raw_data = self.teleop_interface.advance()

        reset = False

        if teleoperation_active and not reset_stopped_teleoperation:

            return self.step_active_teleoperation(teleop_data, raw_data)

        else:

            self.env.sim.render()

        if reset_stopped_teleoperation:
            _, _, _, intructions = self.step_reset_teleoperation()
            reset = True
        self.num_frame += 1

        return None, {}, reset, intructions
