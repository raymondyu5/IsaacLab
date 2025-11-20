import torch
import numpy as np

import copy

import imageio

from scripts.workflows.open_policy.utils.buffer_utils import reset_buffer, update_buffer
import tqdm

from scripts.workflows.hand_manipulation.env.rl_env.bimanual_rl_step_wrapper import BimanulRLStepWrapper

import matplotlib.pyplot as plt


def visualize_latent_space(data_loader, name="train"):

    plt.figure(figsize=(12, 4))
    for i in range(data_loader.shape[1]):
        plt.hist(data_loader[:, i].cpu().numpy(),
                 bins=40,
                 alpha=0.6,
                 label=f"z[{i}]")
    plt.title(f"{name.capitalize()} Latent Distribution per Dimension")
    plt.xlabel("Latent Value")
    plt.ylabel("Frequency")
    plt.legend()

    plt.show()


class BimanualRLDatawrapperEnv(BimanulRLStepWrapper):

    def __init__(
        self,
        env,
        env_config,
        args_cli,
        use_relative_pose=False,
    ):

        self.env = env.env
        self.device = self.env.device
        self.args_cli = args_cli

        self.use_relative_pose = use_relative_pose
        self.env_config = env_config
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        super().__init__(args_cli, env_config, env)
        self.init_setting()
        if self.args_cli.add_right_hand:
            self.hand_side = "right"
        elif self.args_cli.add_left_hand:
            self.hand_side = "left"
        self.eval_success = []
        self.rollout_reward = []
        self.eval_iter = 0

    def process_dict_obs(self, obs):

        proccess_action = []
        for key, value in obs["policy"].items():
            proccess_action.append(value)

        return torch.cat(proccess_action, dim=1)

    def eval_checkpoint(self, agent, last_obs):
        reset_buffer(self)

        last_obs, _ = self.reset()

        rewards_list = []
        for i in range(180):
            if isinstance(last_obs["policy"], dict):
                proccess_last_obs = self.process_dict_obs(last_obs)
            else:
                proccess_last_obs = last_obs["policy"]

            actions, _ = agent.predict(proccess_last_obs.cpu().numpy(),
                                       deterministic=True)

            next_obs, rewards, terminated, time_outs, extras, clip_actions = self.step(
                torch.as_tensor(actions).to(self.device))

            last_obs = copy.deepcopy(next_obs)
            rewards_list.append(rewards.unsqueeze(0))

            update_buffer(self, next_obs, last_obs, clip_actions, rewards,
                          terminated, time_outs)

        rewards_list = torch.concat(rewards_list, dim=0)
        # self.plot_reward(rewards_list)

        if self.args_cli.save_path is not None:
            self.collector_interface.add_demonstraions_to_buffer(
                self.obs_buffer,
                self.action_buffer,
                self.rewards_buffer,
                self.does_buffer,
            )
        # visualize_latent_space(next_obs["policy"])

        return last_obs["policy"][
            f"{ self.hand_side }_manipulated_object_pose"][:, 2] > 0.3

    def plot_reward(self, rewards_list):

        # rewards_list: shape (timesteps, num_envs)
        timesteps, num_envs = rewards_list.shape
        mean_rewards = rewards_list.mean(dim=1)  # shape: (timesteps,)

        plt.figure(figsize=(12, 6))

        # Plot individual environment rewards
        for i in range(num_envs):
            plt.plot(rewards_list[:, i].cpu().numpy(),
                     color='gray',
                     alpha=0.3,
                     linewidth=1)

        # Plot the mean reward
        plt.plot(mean_rewards.cpu().numpy(),
                 color='blue',
                 label='Mean Reward',
                 linewidth=2)

        plt.title("Reward over Time")
        plt.xlabel("Timestep")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def eval_all_checkpoint(self, agent, last_obs, rl_agent_env):
        reset_buffer(self)

        last_obs, _ = self.reset()
        # self.eval_iter = 20

        agent = agent.load(self.args_cli.checkpoint +
                           f"model_{int(self.eval_iter*20)}.zip",
                           rl_agent_env,
                           print_system_info=True)

        reward = 0

        for i in range(180):
            if isinstance(last_obs["policy"], dict):
                proccess_last_obs = self.process_dict_obs(last_obs)
            else:
                proccess_last_obs = last_obs["policy"]

            actions, _ = agent.predict(proccess_last_obs.cpu().numpy(),
                                       deterministic=True)

            next_obs, rewards, terminated, time_outs, extras, clip_actions = self.step(
                torch.as_tensor(actions).to(self.device))

            last_obs = copy.deepcopy(next_obs)

            update_buffer(self, next_obs, last_obs, clip_actions, rewards,
                          terminated, time_outs)
            reward += rewards.sum().item() / self.env.num_envs

        if self.args_cli.save_path is not None:
            self.collector_interface.add_demonstraions_to_buffer(
                self.obs_buffer,
                self.action_buffer,
                self.rewards_buffer,
                self.does_buffer,
            )
        # visualize_latent_space(next_obs["policy"])
        self.eval_iter += 1
        success = last_obs["policy"][
            f"{ self.hand_side }_manipulated_object_pose"][:, 2] > 0.3
        self.eval_success.append(success.sum().item() / self.env.num_envs)
        self.rollout_reward.append(reward)

        # np.save(
        #     f"{self.args_cli.log_dir}/eval_success.npy",
        #     np.array(self.eval_success),
        # )
        # np.save(
        #     f"{self.args_cli.log_dir}/rollout_reward.npy",
        #     np.array(self.rollout_reward),
        # )

        np.savez(
            f"{self.args_cli.log_dir}/{self.hand_side}_hand_success",
            np.array(self.eval_success),
        )
        np.savez(
            f"{self.args_cli.log_dir}/{self.hand_side}_hand_rollout",
            np.array(self.rollout_reward),
        )

        return success

    def eval_symmetry(self, agent, last_obs):
        reset_buffer(self)

        last_obs, _ = self.reset()
        left_object_pose = last_obs["policy"]["left_manipulated_object_pose"]
        right_object_pose = left_object_pose.clone()
        right_object_pose[..., :3] += self.env.scene.env_origins
        right_object_pose[:, 1] -= 0.5
        self.env.scene["right_hand_object"].write_root_pose_to_sim(
            right_object_pose[:, :7],
            torch.arange(self.env.num_envs).to(self.device))

        total_frame = 80  #self.horizon
        i = 0

        for i in range(180):
            if isinstance(last_obs["policy"], dict):
                proccess_last_obs = self.process_dict_obs(last_obs)
            else:
                proccess_last_obs = last_obs["policy"]

            actions, _ = agent.predict(proccess_last_obs.cpu().numpy()[
                ..., :int(proccess_last_obs.shape[-1] / 2)],
                                       deterministic=True)
            actions *= 0.0
            # actions[:, 6:] = np.random.uniform(
            #     low=-1.0,
            #     high=1.0,
            #     size=(actions.shape[0], actions.shape[1] - 6),
            # )
            right_actions = actions.copy()

            actions = np.concatenate([actions, right_actions], axis=-1)

            next_obs, rewards, terminated, time_outs, extras, clip_actions = self.step(
                torch.as_tensor(actions).to(self.device))

            last_obs = copy.deepcopy(next_obs)

            update_buffer(self, next_obs, last_obs, clip_actions, rewards,
                          terminated, time_outs)

        if self.args_cli.save_path is not None:
            self.collector_interface.add_demonstraions_to_buffer(
                self.obs_buffer,
                self.action_buffer,
                self.rewards_buffer,
                self.does_buffer,
            )
        # visualize_latent_space(next_obs["policy"])

        return last_obs["policy"][
            f"{ self.hand_side }_manipulated_object_pose"][:, 2] > 0.3
