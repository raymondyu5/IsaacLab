import torch

import numpy as np

from scripts.workflows.hand_manipulation.utils.visionpro_utils import *

from scripts.workflows.hand_manipulation.env.teleop_env.bunny_env import BunnyEnv


class BunnyVisionProEnv(BunnyEnv):

    def __init__(self, env, args_cli, save_config, render_viser=False):
        super().__init__(env, args_cli, save_config, render_viser)

        if self.args_cli.free_hand:
            self.run = self.run_free_hand
        else:
            self.run = self.run_arm_hand

    def run_arm_hand(self):
        if self.args_cli.proccess_raw:
            self.run_raw_arm_hand()
        else:
            self.run_bunny_arm_hand()

    def run_bunny_arm_hand(self):
        self.env.reset()
        for i in range(500):
            teleop_cmd = self.teleop_client.get_teleop_cmd()
            right_action = torch.tensor(teleop_cmd[1], device=self.env.device)
            left_action = torch.tensor(teleop_cmd[0], device=self.env.device)

            if self.play_mode:
                actions = []

                if self.args_cli.add_left_hand:

                    left_hand_action = left_action[self.left_retarget2isaac]

                    actions.append(left_hand_action)

                if self.args_cli.add_right_hand:
                    right_hand_action = right_action[self.right_retarget2isaac]

                    right_hand_action[..., -16:] = 0.0
                    actions.append(right_hand_action)
                actions = torch.cat(actions, dim=0).unsqueeze(0)

                obs, rewards, terminated, time_outs, extras = self.env.step(
                    actions)
            else:
                left_action = left_action[self.left_retarget2isaac]
                right_action = right_action[self.right_retarget2isaac]
                self.left_robot.root_physx_view.set_dof_positions(
                    left_action.unsqueeze(0), indices=self.env_ids)
                self.right_robot.root_physx_view.set_dof_positions(
                    right_action.unsqueeze(0), indices=self.env_ids)

                # actions = torch.cat(actions, dim=0).unsqueeze(0)
                obs, rewards, terminated, time_outs, extras = self.env.step(
                    torch.as_tensor(self.env.action_space.sample() * 0.0).to(
                        self.env.device))
            if self.render_viser:
                self.livestream.update(update_camera=False)
