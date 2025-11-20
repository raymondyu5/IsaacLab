# needed to import for allowing type-hinting: torch.Tensor | dict[str, torch.Tensor]
from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn  # noqa: F401
from typing import Any

from scripts.workflows.hand_manipulation.utils.action_utils import contruction_space
from stable_baselines3.common.utils import constant_fn
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn

# obs_keys = [
#     'right_hand_joint_pos', 'right_ee_pose', 'right_target_object_pose',
#     'right_contact_obs', 'right_object_in_tip', 'right_manipulated_object_pose'
# ]

obs_keys = [
    'right_hand_joint_pos', 'right_ee_pose', 'right_object_in_tip',
    'right_manipulated_object_pose'
]
import cv2


class StateCustomSerlEnvWrapper:
    """Custom environment wrapper for handling environments in a specific way."""

    def __init__(self):
        """Initialize the wrapper.

        Args:
            env: The environment to wrap around.
            concatenate_obs: Whether to concatenate observations into a single array.
        """

        self.observation_space = gym.spaces.Box(
            low=-np.inf,  # or np.full(52, -1.0)
            high=np.inf,  # or np.full(52,  1.0)
            shape=(62, ),
            dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0,
                                           high=1.0,
                                           shape=(22, ),
                                           dtype=np.float32)


class ImageCustomSerlEnvWrapper:
    """Custom environment wrapper for handling environments in a specific way."""

    def __init__(self):
        """Initialize the wrapper.

        Args:
            env: The environment to wrap around.
            concatenate_obs: Whether to concatenate observations into a single array.
        """

        self.observation_space = gym.spaces.Dict({
            # your old state vector
            "state":
            gym.spaces.Box(low=-np.inf,
                           high=np.inf,
                           shape=(45, ),
                           dtype=np.float32),
            # the RGB image
            "rgb":
            gym.spaces.Box(
                low=0,
                high=255,
                shape=(224, 224, 3),  # e.g. (224, 224, 3)
                dtype=np.uint8)
        })
        self.action_space = gym.spaces.Box(low=-1.0,
                                           high=1.0,
                                           shape=(22, ),
                                           dtype=np.float32)


class Sb3EnvWrapper(VecEnv):

    def __init__(self, env):
        """Initialize the wrapper.

        Args:
            env: The environment to wrap around.

        Raises:
            ValueError: When the environment is not an instance of :class:`ManagerBasedRLEnv` or :class:`DirectRLEnv`.
        """

        # check that input is valid
        from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
        if not isinstance(env.env.unwrapped,
                          ManagerBasedRLEnv) and not isinstance(
                              env.env.unwrapped, DirectRLEnv):
            raise ValueError(
                "The environment must be inherited from ManagerBasedRLEnv or DirectRLEnv. Environment type:"
                f" {type(env)}")
        # initialize the wrapper

        self.env = env

        # collect common information
        self.num_envs = self.unwrapped.num_envs
        self.sim_device = self.unwrapped.device
        self.render_mode = self.unwrapped.render_mode

        if getattr(env, "use_last_action", False):

            self.unwrapped.observation_space["policy"][
                "last_action"] = gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(
                        1,
                        self.env.last_diffusion_action_dim,
                    ),
                    dtype=np.float32)

        self.concatenate_obs, observation_space, action_space = contruction_space(
            self, self.env, obs_keys, sqqueeze_first_dim=True)

        VecEnv.__init__(self, self.num_envs, observation_space, action_space)

        self._ep_rew_buf = torch.zeros(self.num_envs, device=self.sim_device)
        self._ep_len_buf = torch.zeros(self.num_envs, device=self.sim_device)

    def __str__(self):
        """Returns the wrapper name and the :attr:`env` representation string."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

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

    @classmethod
    def class_name(cls) -> str:
        """Returns the class name of the wrapper."""
        return cls.__name__

    @property
    def unwrapped(self):
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """

        return self.env.env.unwrapped

    def get_episode_rewards(self) -> list[float]:
        """Returns the rewards of all the episodes."""
        return self._ep_rew_buf.cpu().tolist()

    def get_episode_lengths(self) -> list[int]:
        """Returns the number of time-steps of all the episodes."""
        return self._ep_len_buf.cpu().tolist()

    def seed(self, seed: int | None = None) -> list[int | None]:  # noqa: D102
        return [self.unwrapped.seed(seed)] * self.unwrapped.num_envs

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

    def set_attr(self, attr_name, value, indices=None):  # noqa: D102
        raise NotImplementedError("Setting attributes is not supported.")

    def reset(self):

        self.reset_obs_dict, _, base_actions = self.env.reset()
        next_obs = self._process_obs(self.reset_obs_dict,
                                     concatenate_obs=self.concatenate_obs)

        self._ep_rew_buf.zero_()
        self._ep_len_buf.zero_()
        base_actions = base_actions.detach().cpu().numpy()
        return next_obs  #, base_actions

    def step_async(self, actions):  # noqa: D102
        # convert input to numpy array
        if not isinstance(actions, torch.Tensor):
            actions = np.asarray(actions)
            actions = torch.from_numpy(actions).to(device=self.sim_device,
                                                   dtype=torch.float32)
        else:
            actions = actions.to(device=self.sim_device, dtype=torch.float32)
        self._async_actions = actions.reshape(self.num_envs, -1)

    def step_wait(self):
        # record step information

        rollouts = self.env.step(self._async_actions)
        # print(action[..., :3])  # Debug: print first 3 action valuess

        next_obs_dict, rew, terminated, truncated, extras, base_actions = rollouts[:
                                                                                   6]

        self._ep_rew_buf += rew
        self._ep_len_buf += 1

        # assert not np.isnan(rew).any(), "NaN values found in reward!"
        dones = terminated | truncated
        reset_ids = (dones > 0).nonzero(as_tuple=False)

        next_obs = self._process_obs(next_obs_dict,
                                     concatenate_obs=self.concatenate_obs)

        rew = rew.detach().cpu().numpy()
        terminated = terminated.detach().cpu().numpy()
        truncated = truncated.detach().cpu().numpy()
        dones = dones.detach().cpu().numpy()

        infos = self._process_extras(next_obs, terminated, truncated, extras,
                                     reset_ids)

        assert not np.isnan(rew).any(), "NaN in reward!"
        assert not np.isnan(terminated).any(), "NaN in terminated!"
        assert not np.isnan(truncated).any(), "NaN in truncated!"
        assert not np.isnan(dones).any(), "NaN in dones!"

        if dones[0]:

            next_obs = self.reset()

        if isinstance(next_obs, dict):
            for k, v in next_obs.items():
                assert not np.isnan(v).any(), f"NaN in next_obs[{k}]!"
        else:
            assert not np.isnan(next_obs).any(), "NaN in next_obs!"

        return next_obs, rew, dones, infos  #, base_action

    def close(self):  # noqa: D102
        self.env.close()

    def _process_obs(
            self,
            obs_dict: torch.Tensor | dict[str, torch.Tensor],
            concatenate_obs: bool = True
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Convert observations into NumPy data type."""
        # Sb3 doesn't support asymmetric observation spaces, so we only use "policy"

        obs = obs_dict["policy"]

        if getattr(self.env, "use_last_action", False):
            obs_keys.append("last_action")

        if not concatenate_obs and isinstance(obs, dict):
            dict_obs = dict()
            low_dim_obs = []

            for key, value in obs.items():

                if key not in obs_keys:
                    continue
                if "rgb" in key:

                    dict_obs[key] = cv2.resize(
                        value.squeeze(0).squeeze(0).cpu().numpy(),
                        (224, 224))[None]
                else:
                    low_dim_obs.append(value.squeeze(0).cpu().numpy())
            low_dim_obs = np.concatenate(low_dim_obs, axis=-1)
            assert not np.isnan(
                low_dim_obs).any(), "NaN values found in low_dim_obs!"
            dict_obs["state"] = low_dim_obs
            # dict_obs["images"] = image_obs
            return dict_obs

        # note: ManagerBasedRLEnv uses torch backend (by default).
        if isinstance(obs, dict):
            obs_buffer = []
            for key, value in obs.items():
                if key not in obs_keys:
                    continue
                obs_buffer.append(value.squeeze(0))

            obs = torch.cat(obs_buffer, dim=-1).detach().cpu().numpy()
        elif isinstance(obs, torch.Tensor):

            obs = obs.detach().squeeze(0).cpu().numpy()
        else:
            raise NotImplementedError(f"Unsupported data type: {type(obs)}")

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

    def env_is_wrapped(self, wrapper_class, indices=None):  # noqa: D102
        raise NotImplementedError(
            "Checking if environment is wrapped is not supported.")

    def get_images(self):  # noqa: D102
        raise NotImplementedError("Getting images is not supported.")
