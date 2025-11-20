import numpy as np
from gymnasium import spaces
import gymnasium as gym
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape

import torch as th
from typing import TYPE_CHECKING, Any, Callable, NamedTuple


class RolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor


class RolloutBufferNumpySamples(NamedTuple):
    observations: np.ndarray
    actions: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    rewards: np.ndarray


class OnlineBCBuffer:

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminates: np.ndarray

    # advantages: np.ndarray
    # returns: np.ndarray
    # episode_starts: np.ndarray

    def __init__(self,
                 env,
                 buffer_size,
                 observation_space: spaces.Space,
                 action_space: spaces.Space,
                 device: str,
                 n_envs: int = 1):
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        # type: ignore[assignment]

        self.n_envs = n_envs

        self.action_dim = get_action_dim(action_space)
        self.env = env

        if isinstance(self.observation_space, gym.spaces.Dict):
            flattened_space = gym.spaces.utils.flatten_space(
                self.observation_space)

            # Get total observation dimension
            obs_dim = np.prod(flattened_space.shape)

            # Define new observation space
            num_envs = self.env.num_envs  # Adjust based on your setting
            self.obs_shape = int(obs_dim / num_envs)
        else:

            self.obs_shape = self.observation_space.shape[1]

        self.device = device

        self.reset()

    def reset(self) -> None:

        self.observations = np.zeros(
            (self.buffer_size, self.n_envs, self.obs_shape), dtype=np.float32)
        self.next_observations = np.zeros(
            (self.buffer_size, self.n_envs, self.obs_shape), dtype=np.float32)
        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs),
                                dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs),
                                       dtype=np.float32)
        self.terminates = np.zeros((self.buffer_size, self.n_envs),
                                   dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs),
                              dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        terminates: np.ndarray,
        dones: np.ndarray,
        # episode_start: np.ndarray,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """

        self.observations = np.concatenate(obs, axis=0)
        self.next_observations = np.concatenate(next_obs, axis=0)
        self.actions = np.concatenate(action, axis=0)
        self.rewards = np.concatenate(reward, axis=0)
        self.terminates = np.concatenate(terminates, axis=0)
        self.dones = np.concatenate(dones, axis=0)

        self.buffer_size = len(self.observations)
        # self.episode_starts = np.array(episode_start)

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def get(self, batch_size):

        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(
                    self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx:start_idx + batch_size])
            start_idx += batch_size

    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data (may be useful to avoid changing things
            by reference). This argument is inoperative if the device is not the CPU.
        :return:
        """
        if copy:
            return th.tensor(array, device=self.device)
        return th.as_tensor(array, device=self.device)

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: int = 0,
    ):
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.next_observations[batch_inds],
            self.dones[batch_inds],
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))

    def sample(self, batch_size: int, env=None, use_numpy=False):
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """

        indices = np.random.choice(self.buffer_size, size=batch_size)

        data = (
            self.observations[indices],
            self.actions[indices],
            self.next_observations[indices],
            self.dones[indices][:, None],
            self.rewards[indices][:, None],
        )
        if use_numpy:
            return RolloutBufferNumpySamples(*tuple(data))
        else:
            return RolloutBufferSamples(*tuple(map(self.to_torch, data)))
