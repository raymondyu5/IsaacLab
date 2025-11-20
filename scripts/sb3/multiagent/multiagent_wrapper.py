import numpy as np
import torch

from scripts.sb3.multiagent.IPPO import IPPO
from scripts.sb3.multiagent.MAPPO import MAPPO


class MultiAgentWrapper:

    def __init__(self,
                 env,
                 num_agents,
                 policy_arch,
                 args_cli,
                 eval=False,
                 eval_per_step=True,
                 **agent_cfg):
        self.env = env
        self.num_agents = num_agents
        self.args_cli = args_cli
        self.eval_iter = 0
        self.eval_per_step = eval_per_step
        self.left_hand_success = []
        self.right_hand_success = []

        self.left_hand_rollout = []
        self.right_hand_rollout = []

        if args_cli.rl_type == "mappo":

            self.agent = MAPPO(policy_arch,
                               env,
                               verbose=1,
                               gpu_buffer=False,
                               args_cli=args_cli,
                               **agent_cfg)
            if eval:
                self.prepare_rollout = self.prepare_rollout_mappo

                self.load = self.load_mappo

        elif args_cli.rl_type == "ippo":
            self.agent = IPPO(policy_arch,
                              env,
                              verbose=1,
                              args_cli=args_cli,
                              **agent_cfg)
            if eval:
                self.prepare_rollout = self.prepare_rollout_ippo
                self.load = self.load_ippo

    def load_ippo(self, path):

        if self.eval_per_step:
            path = path + f"_{int(self.eval_iter*20)}"
            self.eval_iter += 1

        if self.args_cli.add_left_hand:

            saved_variables = torch.load(path + f"_left.zip",
                                         map_location=torch.device("cuda"),
                                         weights_only=False)
            self.agent.policies[0].policy.load_state_dict(
                saved_variables["state_dict"])
        if self.args_cli.add_right_hand:
            saved_variables = torch.load(path + f"_right.zip",
                                         map_location=torch.device("cuda"),
                                         weights_only=False)
            self.agent.policies[-1].policy.load_state_dict(
                saved_variables["state_dict"])

    def load_mappo(self, path, load_single_agent=True):
        name = ["left", "right"]
        if self.eval_per_step:
            path = path + f"_{int(self.eval_iter*20)}"
            self.eval_iter += 1

        for index, policy in enumerate(self.agent.policies):

            saved_variables = torch.load(path + f"_{name[index]}.zip",
                                         map_location=torch.device("cuda"),
                                         weights_only=False)
            self.agent.policies[index].policy.load_state_dict(
                saved_variables["state_dict"])

    def learn(self, total_timesteps, callback):
        self.agent.learn(total_timesteps=total_timesteps, callback=callback)

    def prepare_rollout_ippo(self):

        print("=======================")
        import time
        start = time.time()
        if self.eval_per_step:
            self.load(self.args_cli.checkpoint)
            print("iteration: ", int(self.eval_iter * 20))

        last_obs = self.env.reset()

        left_rewards = torch.zeros(
            self.env.num_envs,
            device=self.env.unwrapped.device,
        )
        right_rewards = torch.zeros(
            self.env.num_envs,
            device=self.env.unwrapped.device,
        )

        for i in range(180):
            actions = []
            if self.args_cli.add_left_hand:
                actions.append(self.agent.policies[0].policy.predict(
                    last_obs[:, 0], deterministic=True)[0])
                left_rewards += self.env.env.env.reward_manager._episode_reward[
                    "left_rewards"].clone()
            if self.args_cli.add_right_hand:
                actions.append(self.agent.policies[-1].policy.predict(
                    last_obs[:, -1], deterministic=False)[0])
                right_rewards += self.env.env.env.reward_manager._episode_reward[
                    "right_rewards"].clone()

            actions = np.concatenate(actions, axis=1)
            last_obs, shared_obs, rewards, dones, infos = self.env.step(
                torch.as_tensor(actions).to("cuda"))
        print("time: ", time.time() - start)

        if self.args_cli.add_left_hand:
            success_or_not = self.env.env.env.scene[
                "left_hand_object"]._data.root_state_w.clone()[:, 2] > 0.3

            self.left_hand_success.append(success_or_not.sum().item() /
                                          self.env.num_envs)

            self.left_hand_rollout.append(left_rewards.sum().item() /
                                          self.env.num_envs)

            np.savez(f"{self.args_cli.log_dir}/left_hand_success",
                     self.left_hand_success)

            np.savez(f"{self.args_cli.log_dir}/left_hand_rollout",
                     self.left_hand_rollout)

            print(
                f"left hand success: {self.left_hand_success[-1]} left hand rollout: {self.left_hand_rollout[-1]}"
            )

        if self.args_cli.add_right_hand:
            success_or_not = self.env.env.env.scene[
                "right_hand_object"]._data.root_state_w.clone()[:, 2] > 0.3
            self.right_hand_success.append(success_or_not.sum().item() /
                                           self.env.num_envs)
            self.right_hand_rollout.append(right_rewards.sum().item() /
                                           self.env.num_envs)

            np.savez(f"{self.args_cli.log_dir}/right_hand_success",
                     self.right_hand_success)
            np.savez(f"{self.args_cli.log_dir}/right_hand_rollout",
                     self.right_hand_rollout)
            print(
                f"right hand success: {self.right_hand_success[-1]} right hand rollout: {self.right_hand_rollout[-1]}"
            )

    def prepare_rollout_mappo(self):
        print("=======================")
        import time
        start = time.time()
        if self.eval_per_step:
            self.load(self.args_cli.checkpoint)
            print("iteration: ", int(self.eval_iter * 20))

        last_obs = self.env.reset()
        left_rewards = torch.zeros(
            self.env.num_envs,
            device=self.env.unwrapped.device,
        )
        right_rewards = torch.zeros(
            self.env.num_envs,
            device=self.env.unwrapped.device,
        )

        for i in range(180):
            actions = []
            if self.args_cli.add_left_hand:
                actions.append(self.agent.policies[0].policy.predict(
                    last_obs[:, 0], deterministic=True)[0])
                left_rewards += self.env.env.env.reward_manager._episode_reward[
                    "left_rewards"].clone()
            if self.args_cli.add_right_hand:
                actions.append(self.agent.policies[-1].policy.predict(
                    last_obs[:, -1], deterministic=True)[0])
                right_rewards += self.env.env.env.reward_manager._episode_reward[
                    "right_rewards"].clone()

            actions = np.concatenate(actions, axis=1)
            last_obs, shared_obs, rewards, dones, infos = self.env.step(
                torch.as_tensor(actions).to("cuda"))
        print("time: ", time.time() - start)

        if self.args_cli.add_left_hand:
            success_or_not = self.env.env.env.scene[
                "left_hand_object"]._data.root_state_w.clone()[:, 2] > 0.3

            self.left_hand_success.append(success_or_not.sum().item() /
                                          self.env.num_envs)

            self.left_hand_rollout.append(left_rewards.sum().item() /
                                          self.env.num_envs)

            np.savez(f"{self.args_cli.log_dir}/left_hand_success",
                     self.left_hand_success)

            np.savez(f"{self.args_cli.log_dir}/left_hand_rollout",
                     self.left_hand_rollout)

            print(
                f"left hand success: {self.left_hand_success[-1]} left hand rollout: {self.left_hand_rollout[-1]}"
            )

        if self.args_cli.add_right_hand:
            success_or_not = self.env.env.env.scene[
                "right_hand_object"]._data.root_state_w.clone()[:, 2] > 0.3
            self.right_hand_success.append(success_or_not.sum().item() /
                                           self.env.num_envs)
            self.right_hand_rollout.append(right_rewards.sum().item() /
                                           self.env.num_envs)

            np.savez(f"{self.args_cli.log_dir}/right_hand_success",
                     self.right_hand_success)
            np.savez(f"{self.args_cli.log_dir}/right_hand_rollout",
                     self.right_hand_rollout)
            print(
                f"right hand success: {self.right_hand_success[-1]} right hand rollout: {self.right_hand_rollout[-1]}"
            )
