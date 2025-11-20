from scripts.serl.env.residual_env import ResidualEnv
import os
import matplotlib.pyplot as plt

import torch
import copy

from scripts.rsrl.env.residual_eval_wrapper import ResidualEvalRLWrapper


class ResidualRLDatawrapperEnv(ResidualEnv):

    def __init__(
        self,
        env,
        env_config,
        args_cli,
        use_relative_pose=False,
        obs_keys=[],
        eval_mode=False,
    ):
        super().__init__(
            env,
            env_config,
            args_cli,
            use_relative_pose=use_relative_pose,
        )
        self.obs_keys = obs_keys

        if eval_mode:

            self.eval_env = ResidualEvalRLWrapper(env,
                                                  env_config,
                                                  args_cli,
                                                  hand_side=self.hand_side)

            self.init_eval_result_folder = self.eval_env.init_eval_result_folder

            self.eval_checkpoint = self.eval_env.eval_checkpoint
            self.eval_all_checkpoint = self.eval_env.eval_all_checkpoint
            setattr(self.eval_env, "reset", self.reset)
            setattr(self.eval_env, "step", self.step)
            setattr(self.eval_env, "_process_obs", self._process_obs)

    def process_dict_obs(self, obs):

        proccess_action = []
        for key, value in obs["policy"].items():
            if key in self.obs_keys:
                proccess_action.append(value)

        return torch.cat(proccess_action, dim=1)

    def warmup(self, agent, bc_eval=True):
        with torch.no_grad():

            last_obs, _ = self.reset()

            rewards = 0.0

            while True:
                if isinstance(last_obs["policy"], dict):
                    proccess_last_obs = self.process_dict_obs(last_obs)
                else:
                    proccess_last_obs = last_obs["policy"]

                actions, _ = agent.predict(proccess_last_obs.cpu().numpy(),
                                           deterministic=True)

                # step the environment

                next_obs, rewards, terminated, time_outs, extras, _ = self.step(
                    actions, bc_eval)
                rewards += rewards.sum().item()
                last_obs = copy.deepcopy(next_obs)
                dones = terminated | time_outs
                self.last_finger_pose = self.env.scene[
                    f"{self.hand_side}_hand"].data.joint_pos[...,
                                                             -16:].clone()

                if self.env.episode_length_buf[
                        0] == self.env.max_episode_length - 1:

                    success = self.evaluate_success()

                if dones[0]:
                    break

        print('====================================')
        print('====================================')
        print('====================================')
        print("Warmup done. Success rate: ",
              success.sum().item() / self.env.num_envs)
        print('====================================')
        print('====================================')
        print('====================================')

        self.env.reset()

        return success.sum().item() / self.env.num_envs
