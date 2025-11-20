# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Wrapper to configure a :class:`ManagerBasedRLEnv` or :class:`DirectRLEnv` instance to Stable-Baselines3 vectorized environment.

The following example shows how to wrap an environment for Stable-Baselines3:

.. code-block:: python

    from isaaclab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper

    env = Sb3VecEnvWrapper(env)

"""

# needed to import for allowing type-hinting: torch.Tensor | dict[str, torch.Tensor]
from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn  # noqa: F401
from typing import Any

from scripts.workflows.hand_manipulation.env.rl_env.torch_layers import PointNetStateExtractor
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from gym.spaces import Dict, Box
"""
Vectorized environment wrapper.
"""


class MultiAgentEnvWrapper:

    def __init__(self, env, args_cli):

        # check that input is valid
        if not isinstance(env.env.unwrapped,
                          ManagerBasedRLEnv) and not isinstance(
                              env.env.unwrapped, DirectRLEnv):
            raise ValueError(
                "The environment must be inherited from ManagerBasedRLEnv or DirectRLEnv. Environment type:"
                f" {type(env)}")
        # initialize the wrapper

        self.env = env
        self.device = env.env.unwrapped.device
        self.args_cli = args_cli

        # collect common information
        self.num_envs = self.unwrapped.num_envs
        self.sim_device = self.unwrapped.device
        self.render_mode = self.unwrapped.render_mode

        action_space = self.unwrapped.single_action_space

        if isinstance(action_space,
                      gym.spaces.Box) and not action_space.is_bounded("both"):
            action_space = gym.spaces.Box(low=-1,
                                          high=1,
                                          shape=action_space.shape)

        if isinstance(env.observation_space,
                      gym.spaces.Dict) and not env.args_cli.use_dict_obs:
            flattened_space = gym.spaces.utils.flatten_space(
                env.observation_space)

            # Get total observation dimension
            obs_dim = np.prod(flattened_space.shape)

            # Define new observation space
            num_envs = self.env.env.num_envs  # Adjust based on your setting
            observation_space = gym.spaces.Box(
                -np.inf,
                np.inf,
                shape=(int(obs_dim / num_envs), ),  # Ensure shape is a tuple
                dtype=np.float32)
        elif isinstance(env.observation_space,
                        gym.spaces.Dict) and env.args_cli.use_dict_obs:

            num_envs = self.env.env.num_envs  # Adjust based on your setting
            raw_observation_space = env.observation_space["policy"]
            observation_space = {}
            for key, value in raw_observation_space.spaces.items():
                assert isinstance(value,
                                  gym.Space), f"{key} is not a gym.Space"
                assert hasattr(value, "shape"), f"{key} has no shape attribute"

                observation_space[key] = gym.spaces.Box(
                    -np.inf, np.inf, shape=(value.shape[1:]), dtype=np.float32)

                # Now wrap'

            observation_space = gym.spaces.Dict(observation_space)
            self.concatenate_obs = False
        else:
            observation_space = self.unwrapped.single_observation_space[
                "policy"]
        self.share_observation_space = []
        self.observation_space = []
        self.action_space = []
        if args_cli.add_left_hand:
            self.share_observation_space.append(observation_space)
            self.action_space.append(action_space)
        if args_cli.add_right_hand:
            self.share_observation_space.append(observation_space)
            self.action_space.append(action_space)
        if args_cli.add_left_hand and args_cli.add_right_hand:
            self.num_agents = 2

            single_agent_obs = gym.spaces.Box(
                -np.inf,
                np.inf,
                shape=(int(observation_space.shape[-1] / 2)),
                dtype=np.float32)
            self.observation_space.append(single_agent_obs)
            self.observation_space.append(single_agent_obs)
        else:
            self.num_agents = 1
            self.observation_space.append(observation_space)

        self.concatenate_obs = True

        # add buffer for logging episodic information
        self._ep_rew_buf = torch.zeros(self.num_envs, device=self.sim_device)
        self._ep_len_buf = torch.zeros(self.num_envs, device=self.sim_device)

    def __str__(self):
        """Returns the wrapper name and the :attr:`env` representation string."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

    """
    Properties -- Gym.Wrapper
    """

    @classmethod
    def class_name(cls) -> str:
        """Returns the class name of the wrapper."""
        return cls.__name__

    @property
    def unwrapped(self) -> ManagerBasedRLEnv | DirectRLEnv:
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """

        return self.env.env.unwrapped

    """
    Properties
    """

    def get_episode_rewards(self) -> list[float]:
        """Returns the rewards of all the episodes."""
        return self._ep_rew_buf.cpu().tolist()

    def get_episode_lengths(self) -> list[int]:
        """Returns the number of time-steps of all the episodes."""
        return self._ep_len_buf.cpu().tolist()

    """
    Operations - MDP
    """

    def seed(self, seed: int | None = None) -> list[int | None]:  # noqa: D102
        return [self.unwrapped.seed(seed)] * self.unwrapped.num_envs

    def reset(self) -> VecEnvObs:  # noqa: D102

        obs_dict, _ = self.env.reset()

        # reset episodic information buffers
        self._ep_rew_buf.zero_()
        self._ep_len_buf.zero_()
        obs = self._process_obs(obs_dict, concatenate_obs=self.concatenate_obs)

        # convert data types to numpy depending on backend
        return obs, obs, None

    def step_async(self, actions):  # noqa: D102
        # convert input to numpy array

        actions = torch.cat(actions, dim=-1)

        actions = actions.to(device=self.sim_device, dtype=torch.float32)
        # convert to tensor
        self._async_actions = actions

    def step_wait(self, actions=None):  # noqa: D102
        # record step information
        self.step_async(actions)

        rollouts = self.env.step(self._async_actions)

        obs_dict, rew, terminated, truncated, extras = rollouts[:5]

        # update episode un-discounted return and length
        self._ep_rew_buf += rew
        self._ep_len_buf += 1
        # compute reset ids
        dones = terminated | truncated
        reset_ids = (dones > 0).nonzero(as_tuple=False)

        # convert data types to numpy depending on backend
        # note: ManagerBasedRLEnv uses torch backend (by default).

        obs = self._process_obs(obs_dict, concatenate_obs=self.concatenate_obs)

        rew = rew.detach()
        terminated = terminated.detach()
        truncated = truncated.detach()
        dones = dones.detach()
        # convert extra information to list of dicts
        infos = self._process_extras(obs, terminated, truncated, extras,
                                     reset_ids)

        # reset info for terminated environments
        self._ep_rew_buf[reset_ids] = 0
        self._ep_len_buf[reset_ids] = 0

        rewards = []
        does_buffer = []

        if self.args_cli.add_left_hand:
            rewards.append(self.env.env.reward_manager.
                           _episode_reward["left_rewards"].unsqueeze(-1))
            does_buffer.append(dones.unsqueeze(-1))
        if self.args_cli.add_right_hand:
            rewards.append(self.env.env.reward_manager.
                           _episode_reward["right_rewards"].unsqueeze(-1))
            does_buffer.append(dones.unsqueeze(-1))

        if dones[0]:

            self.reset()

        return obs, obs, torch.cat(rewards,
                                   dim=1), torch.cat(does_buffer,
                                                     dim=1), infos, None

    def close(self):  # noqa: D102
        self.env.close()

    def get_attr(self, attr_name, indices=None):  # noqa: D102
        # resolve indices
        if indices is None:
            indices = slice(None)
            num_indices = self.num_envs
        else:
            num_indices = len(indices)
        # obtain attribute value
        attr_val = getattr(self.env, attr_name)
        # return the value
        if not isinstance(attr_val, torch.Tensor):
            return [attr_val] * num_indices
        else:
            if self.gpu_buffer:
                return attr_val[indices].detach()
            return attr_val[indices].detach().cpu().numpy()

    def set_attr(self, attr_name, value, indices=None):  # noqa: D102
        raise NotImplementedError("Setting attributes is not supported.")

    def env_method(self,
                   method_name: str,
                   *method_args,
                   indices=None,
                   **method_kwargs):  # noqa: D102
        if method_name == "render":
            # gymnasium does not support changing render mode at runtime
            return self.env.render()
        else:
            # this isn't properly implemented but it is not necessary.
            # mostly done for completeness.
            env_method = getattr(self.env, method_name)
            return env_method(*method_args, indices=indices, **method_kwargs)

    def env_is_wrapped(self, wrapper_class, indices=None):  # noqa: D102
        raise NotImplementedError(
            "Checking if environment is wrapped is not supported.")

    def get_images(self):  # noqa: D102
        raise NotImplementedError("Getting images is not supported.")

    """
    Helper functions.
    """

    def _process_obs(
            self,
            obs_dict: torch.Tensor | dict[str, torch.Tensor],
            concatenate_obs: bool = True
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Convert observations into NumPy data type."""
        # Sb3 doesn't support asymmetric observation spaces, so we only use "policy"

        obs = obs_dict["policy"]
        if not concatenate_obs:
            dict_obs = dict()

            for key, value in obs.items():

                dict_obs[key] = value
            return dict_obs

        # note: ManagerBasedRLEnv uses torch backend (by default).
        if isinstance(obs, dict):

            obs = torch.cat([v for v in obs.values()], dim=-1).detach()
        elif isinstance(obs, torch.Tensor):

            obs = obs.detach()

        else:
            raise NotImplementedError(f"Unsupported data type: {type(obs)}")
        obs = obs.reshape(self.num_envs, self.num_agents, -1)

        return obs

    def _process_extras(self, obs: np.ndarray, terminated: np.ndarray,
                        truncated: np.ndarray, extras: dict,
                        reset_ids: np.ndarray) -> list[dict[str, Any]]:
        """Convert miscellaneous information into dictionary for each sub-environment."""
        # create empty list of dictionaries to fill
        infos: list[dict[str, Any]] = [
            dict.fromkeys(extras.keys()) for _ in range(self.num_envs)
        ]
        # fill-in information for each sub-environment
        # note: This loop becomes slow when number of environments is large.
        for idx in range(self.num_envs):
            # fill-in episode monitoring info
            if idx in reset_ids:
                infos[idx]["episode"] = dict()
                infos[idx]["episode"]["r"] = float(self._ep_rew_buf[idx])
                infos[idx]["episode"]["l"] = float(self._ep_len_buf[idx])
            else:
                infos[idx]["episode"] = None
            # fill-in bootstrap information
            infos[idx][
                "TimeLimit.truncated"] = truncated[idx] and not terminated[idx]
            # fill-in information from extras
            for key, value in extras.items():
                # 1. remap extra episodes information safely
                # 2. for others just store their values
                if key == "log":
                    # only log this data for episodes that are terminated
                    if infos[idx]["episode"] is not None:
                        for sub_key, sub_value in value.items():
                            infos[idx]["episode"][sub_key] = sub_value
                else:
                    infos[idx][key] = value[idx]
            # add information about terminal observation separately
            if idx in reset_ids:
                # extract terminal observations
                if isinstance(obs, dict):
                    terminal_obs = dict.fromkeys(obs.keys())
                    for key, value in obs.items():
                        terminal_obs[key] = value[idx]
                else:
                    terminal_obs = obs[idx]
                # add info to dict
                infos[idx]["terminal_observation"] = terminal_obs
            else:
                infos[idx]["terminal_observation"] = None
        # return list of dictionaries
        return infos
