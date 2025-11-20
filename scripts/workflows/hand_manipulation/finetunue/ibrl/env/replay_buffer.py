import sys

sys.path.append("submodule/stable-baselines3")

from stable_baselines3.common.buffers import ReplayBuffer, DictReplayBuffer
from gymnasium import spaces
import gymnasium as gym
import copy

import torch
import numpy as np
from scripts.workflows.open_policy.utils.buffer_utils import reset_buffer, update_buffer
import zarr
import os

import torch as th
from stable_baselines3.common.type_aliases import (NamedTuple)


def construct_replay_buffer(buffer_size=1000000,
                            n_envs=1,
                            observation_space=None,
                            action_space=None):
    """Construct a replay buffer for RL and BC."""
    if isinstance(observation_space, spaces.Dict):
        replay_buffer = DictReplayBuffer(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            n_envs=n_envs,
            optimize_memory_usage=True,
            handle_timeout_termination=False,
        )
        concatenate_obs = False
    else:
        replay_buffer = ReplayBuffer(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            optimize_memory_usage=True,
            handle_timeout_termination=False,
            n_envs=n_envs,
        )
        concatenate_obs = True

    return replay_buffer, concatenate_obs


class ReplayBufferSamples(NamedTuple):
    obs: th.Tensor
    action: th.Tensor
    next_obs: th.Tensor

    reward: th.Tensor
    # For n-step replay buffer
    bootstrap: None


class RelayBufferWrapper:

    def __init__(self,
                 rl_buffer_size=1000000,
                 bc_buffer_size=1000000,
                 n_envs=1,
                 observation_space=None,
                 action_space=None):

        self.bc_replay_buffer, self.concatenate_obs = construct_replay_buffer(
            bc_buffer_size, 1, observation_space, action_space)
        self.rl_replay_buffer, self.concatenate_obs = construct_replay_buffer(
            rl_buffer_size, n_envs, observation_space, action_space)

    def add_data_to_buffer(
        self,
        obs,
        next_obs,
        action,
        reward,
        done,
        infos,
    ):
        process_obs = self._process_obs({"policy": obs})
        process_next_obs = self._process_obs({"policy": next_obs},
                                             self.concatenate_obs)

        self.rl_replay_buffer.add(
            process_obs,
            process_next_obs,
            action.cpu().numpy(),
            reward.cpu().numpy(),
            done.cpu().numpy(),
            infos,
        )

    def add_data(self, next_obs, last_obs, actions, rewards, done):

        self.obs_buffer.append(
            self._process_obs(last_obs, self.concatenate_obs))

        self.action_buffer.append(actions.cpu())
        self.rewards_buffer.append(rewards)

        self.does_buffer.append(done)

    def _process_obs(
            self,
            obs_dict: torch.Tensor | dict[str, torch.Tensor],
            concatenate_obs: bool = True
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Convert observations into NumPy data type."""
        # Sb3 doesn't support asymmetric observation spaces, so we only use "policy"

        obs = obs_dict["policy"]

        if not concatenate_obs and isinstance(obs, dict):
            dict_obs = dict()

            for key, value in obs.items():
                if isinstance(value, torch.Tensor):

                    dict_obs[key] = value.cpu().numpy()
                elif isinstance(value, np.ndarray):
                    dict_obs[key] = value
            return dict_obs

        # note: ManagerBasedRLEnv uses torch backend (by default).
        if isinstance(obs, dict):
            obs_buffer = []
            for key, value in obs.items():
                if key in ["seg_pc", "rgb"]:
                    continue
                obs_buffer.append(value)
            if isinstance(value, torch.Tensor):
                obs = torch.cat(obs_buffer, dim=-1).detach().cpu().numpy()
            elif isinstance(value, np.ndarray):
                obs = np.concatenate(obs_buffer, axis=-1)
        elif isinstance(obs, torch.Tensor):

            obs = obs.detach().cpu().numpy()
        else:
            raise NotImplementedError(f"Unsupported data type: {type(obs)}")

        return obs

    def save_data_to_buffer(self, index):
        num_horizon = len(self.action_buffer)
        rewards = 0

        for id in index:

            for time_id in range(num_horizon):

                act = self.action_buffer[time_id][id]
                obs = self.obs_buffer[time_id][id]
                rew = self.rewards_buffer[time_id][id]
                done = self.does_buffer[time_id][id]
                self.bc_replay_buffer.add(
                    obs,
                    None,
                    act.cpu().numpy(),
                    rew.cpu().numpy(),
                    done.cpu().numpy(),
                    None,
                )
                rewards += rew.cpu().numpy()

        return rewards

    def warm_up(self,
                diffusion_env,
                reset_function,
                preload_num_data,
                num_warm_up_episode,
                load_pretrained_agent,
                pretrain_num_epoch,
                load_path=None,
                obs_names=[]):
        # warm up stage, fill the replay with some episodes
        # it can either be human demos, or generated by the bc, or purely random

        total_reward = 0
        num_episode = 0
        while num_episode < num_warm_up_episode + preload_num_data:

            # if load_path is None:
            #     obs, _ = reset_function()
            #     reset_buffer(self)

            #     sucess = diffusion_env.close_loop_evaluate()
            #     num_episode += sucess.sum().item()
            #     if num_episode >= num_warm_up_episode:
            #         total_reward += self.save_data_to_buffer(
            #             torch.where(sucess)[0].tolist()
            #             [:int(num_episode - num_warm_up_episode)])
            #     else:
            #         total_reward += self.save_data_to_buffer(
            #             torch.where(sucess)[0].tolist())

            # # elif load_pretrained_agent or pretrain_num_epoch > 0:
            # else:
            #     # the policy has been pretrained/initialized

            num_episode, total_reward, normalize_action_range = self.load_pretrained_data(
                load_path, num_warm_up_episode, preload_num_data, obs_names)
            break
        print(
            "warm up episode: ",
            num_episode,
        )
        assert num_episode>= num_warm_up_episode, \
            f"num_episode {num_episode} < num_warm_up_episode {num_warm_up_episode}"
        return normalize_action_range

    def load_pretrained_data(self, load_path, num_warm_up_episode,
                             preload_num_data, obs_names):

        pretrained_data = os.listdir(load_path)
        sample_files = np.random.choice(pretrained_data,
                                        num_warm_up_episode + preload_num_data,
                                        replace=False)
        total_reward = 0

        for index, file in enumerate(sample_files):
            zarr_path = os.path.join(load_path, file)

            data = zarr.open(zarr_path, mode='r')
            obs_dict = {}

            # Get all keys (recursive, full path)
            keys = []
            data.visititems(lambda k, v: keys.append((k, v)))

            actions = np.array(data['data/actions'])
            for name in obs_names:
                obs_dict[name] = np.array(data[f'data/{name}'])

            demo_obs = self._process_obs({"policy": obs_dict},
                                         self.concatenate_obs)

            rewards = np.array(data['data/rewards'])
            dones = np.array(data['data/dones'])
            num_horizon = actions.shape[0]

            for time_id in range(num_horizon - 1):
                act = actions[time_id]
                obs = demo_obs[time_id]
                rew = rewards[time_id]
                done = dones[time_id]
                next_obs = demo_obs[time_id + 1]

                if index < preload_num_data:

                    self.bc_replay_buffer.add(
                        obs,
                        next_obs,
                        act,
                        rew,
                        done,
                        None,
                    )
                else:
                    self.rl_replay_buffer.add(
                        obs,
                        next_obs,
                        act,
                        rew,
                        done,
                        None,
                    )
                total_reward += rew.sum()

        assert self.bc_replay_buffer.full,\
            f"The buffer is not full"
        act_min, act_max = np.min(self.bc_replay_buffer.actions,
                                  axis=0)[0], np.max(
                                      self.bc_replay_buffer.actions, axis=0)[0]

        self.bc_replay_buffer.actions = (
            ((self.bc_replay_buffer.actions - act_min) /
             (act_max - act_min)) * 2 - 1)
        self.rl_replay_buffer.actions[:self.rl_replay_buffer.pos] = ((
            (self.rl_replay_buffer.actions[:self.rl_replay_buffer.pos] -
             act_min) / (act_max - act_min)) * 2 - 1)
        return len(sample_files), total_reward, [act_min, act_max]

    def sample(self, rl_size=None, bc_size=None, device="cpu"):

        if rl_size is not None:
            rl_batch = self.rl_replay_buffer.sample(rl_size)
        if bc_size is not None:
            bc_batch = self.bc_replay_buffer.sample(bc_size)
        if rl_size is not None and bc_size is not None:
            return self.merge_batch(rl_batch, bc_batch, device)
        elif rl_size is not None:
            return self.process_batch(rl_batch, device)
        elif bc_size is not None:
            return self.process_batch(bc_batch, device)

    def process_batch(self, batch, device="cpu"):
        """Merge RL and BC batch."""

        if not self.concatenate_obs:

            obs = {
                k: torch.as_tensor(batch.observations[k]).to(device)
                for k, v in batch.observations.items()
            }
            next_obs = {
                k: torch.as_tensor(batch.next_observations[k]).to(device)
                for k, v in batch.next_observations.items()
            }
        else:
            obs = torch.as_tensor(batch.observations).to(device)
            next_obs = torch.as_tensor(batch.next_observations).to(device)
            if torch.isnan(obs).any().item():
                import pdb
                pdb.set_trace()
        action = torch.as_tensor(batch.actions).to(device)
        reward = torch.as_tensor(batch.rewards).to(device).reshape(-1)
        bootstrap = torch.ones(
            reward.shape[0],
            dtype=torch.float32,
            device=reward.device,
        )

        if self.concatenate_obs:
            return ReplayBufferSamples(
                obs={"state": obs},
                action={"action": action},
                next_obs={"state": next_obs},
                # dones=torch.cat([batch1.dones, batch0.dones],
                #                 dim=0).to(device),
                reward=reward,
                bootstrap=bootstrap,
            )

    def merge_batch(self, batch0, batch1, device):
        """Merge RL and BC batch."""

        if not self.concatenate_obs:

            obs = {
                k: torch.cat([v, batch0.observations[k]], dim=0).to(device)
                for k, v in batch1.observations.items()
            }
            next_obs = {
                k: torch.cat([v, batch0.next_observations[k]],
                             dim=0).to(device)
                for k, v in batch1.next_observations.items()
            }
        else:
            obs = torch.cat([batch1.observations, batch0.observations],
                            dim=0).to(device)
            next_obs = torch.cat(
                [batch1.next_observations, batch0.next_observations],
                dim=0).to(device)
            if torch.isnan(next_obs).any().item():
                import pdb
                pdb.set_trace()
        action = torch.cat([batch1.actions, batch0.actions], dim=0).to(device)
        reward = torch.cat([batch1.rewards, batch0.rewards],
                           dim=0).to(device).squeeze(-1)
        bootstrap = torch.ones(
            reward.shape[0],
            dtype=torch.float32,
            device=reward.device,
        )

        if self.concatenate_obs:
            return ReplayBufferSamples(
                obs={"state": obs},
                action={"action": action},
                next_obs={"state": next_obs},
                # dones=torch.cat([batch1.dones, batch0.dones],
                #                 dim=0).to(device),
                reward=reward,
                bootstrap=bootstrap,
            )
