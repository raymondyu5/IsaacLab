# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
import torch

from rsl_rl.env import VecEnv

from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv

import numpy as np


class RslRlVecEnvWrapper(VecEnv):
    """Wraps around Isaac Lab environment for RSL-RL library

    To use asymmetric actor-critic, the environment instance must have the attributes :attr:`num_privileged_obs` (int).
    This is used by the learning agent to allocate buffers in the trajectory memory. Additionally, the returned
    observations should have the key "critic" which corresponds to the privileged observations. Since this is
    optional for some environments, the wrapper checks if these attributes exist. If they don't then the wrapper
    defaults to zero as number of privileged observations.

    .. caution::

        This class must be the last wrapper in the wrapper chain. This is because the wrapper does not follow
        the :class:`gym.Wrapper` interface. Any subsequent wrappers will need to be modified to work with this
        wrapper.

    Reference:
        https://github.com/leggedrobotics/rsl_rl/blob/master/rsl_rl/env/vec_env.py
    """

    def __init__(self,
                 env: ManagerBasedRLEnv | DirectRLEnv,
                 clip_actions: float | None = None):
        """Initializes the wrapper.

        Note:
            The wrapper calls :meth:`reset` at the start since the RSL-RL runner does not call reset.

        Args:
            env: The environment to wrap around.
            clip_actions: The clipping value for actions. If ``None``, then no clipping is done.

        Raises:
            ValueError: When the environment is not an instance of :class:`ManagerBasedRLEnv` or :class:`DirectRLEnv`.
        """
        # check that input is valid
        if not isinstance(env.env.unwrapped,
                          ManagerBasedRLEnv) and not isinstance(
                              env.env.unwrapped, DirectRLEnv):
            raise ValueError(
                "The environment must be inherited from ManagerBasedRLEnv or DirectRLEnv. Environment type:"
                f" {type(env)}")
        # initialize the wrapper
        self.env = env
        self.clip_actions = clip_actions

        # store information required by wrapper
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length

        # obtain dimensions of the environment
        self._modify_action_space()
        if hasattr(self.unwrapped, "action_manager"):
            self.num_actions = self.action_space
        else:
            self.num_actions = self.action_space
        if hasattr(self.unwrapped, "observation_manager"):
            self.num_obs = self.unwrapped.observation_manager.group_obs_dim[
                "policy"][0]
        else:
            self.num_obs = gym.spaces.flatdim(
                self.unwrapped.single_observation_space["policy"])
        # -- privileged observations
        if (hasattr(self.unwrapped, "observation_manager") and "critic"
                in self.unwrapped.observation_manager.group_obs_dim):
            self.num_privileged_obs = self.unwrapped.observation_manager.group_obs_dim[
                "critic"][0]
        elif hasattr(self.unwrapped, "num_states"
                     ) and "critic" in self.unwrapped.single_observation_space:
            self.num_privileged_obs = gym.spaces.flatdim(
                self.unwrapped.single_observation_space["critic"])
        else:
            self.num_privileged_obs = 0

        self.concatenate_obs = True
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
        self.observation_space = observation_space

        # modify the action space to the clip range

        # reset at the start since the RSL-RL runner does not call reset
        self.env.reset()

    def __str__(self):
        """Returns the wrapper name and the :attr:`env` representation string."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

    """
    Properties -- Gym.Wrapper
    """

    @property
    def cfg(self) -> object:
        """Returns the configuration class instance of the environment."""
        return self.unwrapped.cfg

    @property
    def render_mode(self) -> str | None:
        """Returns the :attr:`Env` :attr:`render_mode`."""
        return self.env.render_mode

    # @property
    # def observation_space(self) -> gym.Space:
    #     """Returns the :attr:`Env` :attr:`observation_space`."""
    #     return self.env.observation_space

    @property
    def action_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`action_space`."""
        return self.new_action_space.shape[0]

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

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        # """Returns the current observations of the environment."""
        # if hasattr(self.unwrapped, "observation_manager"):
        #     obs_dict = self.unwrapped.observation_manager.compute()
        # else:
        #     obs_dict = self.unwrapped._get_observations()
        # return obs_dict["policy"], {"observations": obs_dict}
        pass

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """The episode length buffer."""
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        """Set the episode length buffer.

        Note:
            This is needed to perform random initialization of episode lengths in RSL-RL.
        """
        self.unwrapped.episode_length_buf = value

    """
    Operations - MDP
    """

    def seed(self, seed: int = -1) -> int:  # noqa: D102
        return self.unwrapped.seed(seed)

    def _process_obs(
            self,
            obs_dict: torch.Tensor | dict[str, torch.Tensor],
            concatenate_obs: bool = True
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Convert observations into NumPy data type."""
        # Sb3 doesn't support asymmetric observation spaces, so we only use "policy"

        obs = obs_dict["policy"]
        if not concatenate_obs:

            return obs
        # note: ManagerBasedRLEnv uses torch backend (by default).
        if isinstance(obs, dict):

            obs = torch.cat([v for v in obs.values()], dim=-1).detach()
        elif isinstance(obs, torch.Tensor):

            obs = obs.detach()
        else:
            raise NotImplementedError(f"Unsupported data type: {type(obs)}")

        return obs

    def reset(self) -> tuple[torch.Tensor, dict]:  # noqa: D102
        # reset the environment
        obs_dict, _ = self.env.reset()
        # return observations
        return self._process_obs(obs_dict), {"observations": obs_dict}

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # clip actions
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions,
                                  self.clip_actions)
        # record step information
        # obs_dict, rew, terminated, truncated, extras = self.env.step(actions)

        rollouts = self.env.step(actions)

        obs_dict, rew, terminated, truncated, extras = rollouts[:5]

        # compute dones for compatibility with RSL-RL
        dones = (terminated | truncated).to(dtype=torch.long)
        # move extra observations to the extras dict
        obs = self._process_obs(obs_dict)
        extras["observations"] = obs_dict
        # move time out information to the extras dict
        # this is only needed for infinite horizon tasks
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated

        # return the step information
        return obs, rew, dones, extras

    def close(self):  # noqa: D102
        return self.env.close()

    """
    Helper functions
    """

    def _modify_action_space(self):
        """Modifies the action space to the clip range."""

        action_space = self.env.action_space

        if isinstance(action_space,
                      gym.spaces.Box) and not action_space.is_bounded("both"):

            action_space = gym.spaces.Box(low=-1,
                                          high=1,
                                          shape=(action_space.shape[-1], ))

        if self.env.args_cli.action_framework is not None:
            framework_action_space = action_space.shape[
                0] - self.env.num_hand_joints + self.env.num_finger_actions
            action_space = gym.spaces.Box(low=-1,
                                          high=1,
                                          shape=np.array(
                                              [framework_action_space]))

        self.env.env.unwrapped.single_action_space = action_space
        self.env.env.unwrapped.action_space = action_space
        self.new_action_space = action_space

        # else:

        #     self.env.unwrapped.single_action_space = gym.spaces.Box(
        #         low=-self.clip_actions,
        #         high=self.clip_actions,
        #         shape=(self.num_actions, ))
        #     self.env.unwrapped.action_space = gym.vector.utils.batch_space(
        #         self.env.unwrapped.single_action_space, self.num_envs)
