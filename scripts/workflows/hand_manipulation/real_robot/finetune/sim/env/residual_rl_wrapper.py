from scripts.workflows.hand_manipulation.real_robot.finetune.sim.env.residual_env import ResidualEnv

import os
import matplotlib.pyplot as plt

import torch
import copy


class ResidualRLDatawrapperEnv(ResidualEnv):

    def __init__(
        self,
        env,
        env_config,
        args_cli,
        use_relative_pose=False,
        diffusion_device="cuda",
        use_residual=True,
        residual_delta=0.2,
        diffuclties_range=10.0,
    ):
        super().__init__(
            env,
            env_config,
            args_cli,
            use_relative_pose=use_relative_pose,
            diffusion_device=diffusion_device,
            use_residual=use_residual,
            diffuclties_range=diffuclties_range,
            residual_delta=residual_delta,
        )

    def init_eval_result_folder(self):
        result_folder = "/".join(
            self.args_cli.checkpoint.split("/")) + f'/eval_results'
        os.makedirs(result_folder, exist_ok=True)
        # num_eval_result = len(os.listdir(result_folder))
        # name = self.args_cli.checkpoint.split("/")[3]
        self.save_result_path = f"{result_folder}"
        # os.makedirs(self.save_result_path, exist_ok=True)

    def eval_all_checkpoint(self, policy, rl_agent_env):

        checkpoints_list = sorted(
            [
                f for f in os.listdir(self.args_cli.checkpoint)
                if f.endswith(".zip")
            ],
            key=lambda x: int(x.split("_")[1].split(".")[0]))
        success_list = []
        reward_list = []
        ckpt_iter_list = []

        for ckpt_path in checkpoints_list:

            self.reset()

            agent = policy.load(self.args_cli.checkpoint + f"/{ckpt_path}",
                                rl_agent_env,
                                print_system_info=True)

            success, rewards = self.eval_checkpoint(agent)
            ckpt_iter = int(ckpt_path.split("_")[1].split(".")[0])
            reward_list.append(rewards / self.env.unwrapped.num_envs)
            print("Success rate: ",
                  success.sum().item() / self.env.unwrapped.num_envs)
            success_list.append(success.sum().item() /
                                self.env.unwrapped.num_envs)
            del agent

            num_step = int(self.env.unwrapped.num_envs * 200 / 1000)
            ckpt_iter_list.append(ckpt_iter)
            plt.plot(ckpt_iter_list, success_list, label="Success Rate")
            plt.xlabel(f"Number of Steps (x{num_step}k)")
            plt.ylabel("Success Rate")
            plt.ylim(0, 1)
            plt.legend()
            plt.savefig(os.path.join(self.save_result_path,
                                     "success_rate.png"))
            plt.clf()
            plt.close()
            plt.plot(ckpt_iter_list, reward_list, label="Average Reward")
            plt.xlabel(f"Number of Steps (x{num_step}k)")
            plt.ylabel("Average Reward")

            plt.legend()
            plt.savefig(
                os.path.join(self.save_result_path, "average_reward.png"))
            plt.clf()
            plt.close()

    def eval_checkpoint(self, agent):
        with torch.no_grad():

            last_obs, _ = self.reset()
            rewards = 0.0

            while True:
                if not self.args_cli.use_visual_obs:
                    proccess_last_obs = self._process_obs(last_obs["policy"])

                    actions = torch.as_tensor(
                        agent.predict(proccess_last_obs.cpu().numpy(),
                                      deterministic=True)[0]).to(self.device)

                else:
                    proccess_last_obs = last_obs["policy"].cpu().numpy()

                    actions = torch.as_tensor(
                        agent.predict(proccess_last_obs,
                                      deterministic=True)[0]).to(self.device)

                next_obs, rew, terminated, time_outs, extras, _ = self.step(
                    actions)
                rewards += rew.sum().item()
                last_obs = copy.deepcopy(next_obs)

                if self.env.unwrapped.episode_length_buf[
                        0] == self.env.unwrapped.max_episode_length - 1:

                    success = self.evaluate_success()
                dones = terminated | time_outs
                if dones[0]:
                    break

        return success, rewards

    def warmup(self, agent, bc_eval=True):
        with torch.no_grad():

            last_obs, _ = self.reset()

            rewards = 0.0

            while True:

                actions = torch.zeros((self.env.unwrapped.action_space.shape),
                                      device=self.device)

                # step the environment

                next_obs, rewards, terminated, time_outs, extras, _ = self.step(
                    actions, bc_eval)
                rewards += rewards.sum().item()
                last_obs = copy.deepcopy(next_obs)
                dones = terminated | time_outs
                self.last_finger_pose = self.env.unwrapped.scene[
                    f"{self.hand_side}_hand"].data.joint_pos[...,
                                                             -16:].clone()

                if self.env.unwrapped.episode_length_buf[
                        0] == self.env.unwrapped.max_episode_length - 1:

                    success = self.evaluate_success()

                if dones[0]:
                    break

        print('====================================')
        print('====================================')
        print('====================================')
        print("Warmup done. Success rate: ",
              success.sum().item() / self.env.unwrapped.num_envs)
        print('====================================')
        print('====================================')
        print('====================================')

        self.env.unwrapped.reset()

        return success.sum().item() / self.env.unwrapped.num_envs
