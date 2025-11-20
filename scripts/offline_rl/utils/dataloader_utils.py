import zarr

from collections import defaultdict

import copy

import numpy as np
import os
import os
import random
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import gym
import numpy as np
import pyrallis
import torch

TensorBatch = List[torch.Tensor]


class ReplayBuffer:

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros((buffer_size, state_dim),
                                   dtype=torch.float32,
                                   device=device)
        self._actions = torch.zeros((buffer_size, action_dim),
                                    dtype=torch.float32,
                                    device=device)
        self._rewards = torch.zeros((buffer_size, 1),
                                    dtype=torch.float32,
                                    device=device)
        self._next_states = torch.zeros((buffer_size, state_dim),
                                        dtype=torch.float32,
                                        device=device)
        self._dones = torch.zeros((buffer_size, 1),
                                  dtype=torch.float32,
                                  device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    def load_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError(
                "Trying to load data into non-empty replay buffer")

        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][...,
                                                                        None])
        self._next_states[:n_transitions] = self._to_tensor(
            data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["dones"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0,
                                    min(self._size, self._pointer),
                                    size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def add_transition(self):
        # Use this method to add new data into the replay buffer during fine-tuning.
        # I left it unimplemented since now we do not do fine-tuning.
        raise NotImplementedError


def populate_data_store_from_zarr(demo_path, obs_keys, num_demos=100):
    demo_file = os.listdir(demo_path)
    obs_buffer = defaultdict(list)
    joint_limits = np.array(
        [[-0.03, 0.03], [-0.03, 0.03], [-0.03, 0.03], [-0.05, 0.05],
         [-0.05, 0.05], [-0.05, 0.05], [-0.314, 2.23], [-0.349, 2.094],
         [-0.314, 2.23], [-0.314, 2.23], [-1.047, 1.047],
         [-0.46999997, 2.4429998], [-1.047, 1.047], [-1.047, 1.047],
         [-0.5059999, 1.8849999], [-1.2, 1.8999999], [-0.5059999, 1.8849999],
         [-0.5059999, 1.8849999], [-0.366, 2.0419998], [-1.34, 1.8799999],
         [-0.366, 2.0419998], [-0.366, 2.0419998]], )
    demo_count = 0
    use_image = False

    for file in demo_file:
        if not file.endswith(".zarr"):
            continue
        zarr_path = os.path.join(demo_path, file)
        data = zarr.open(zarr_path, mode='r')
        per_obs_buffer = defaultdict(list)
        actions = np.array(data['data/actions'])

        per_obs_buffer["actions"].append(copy.deepcopy(actions)[:-1])

        rewards = np.array(data['data/rewards'])[:-1]
        per_obs_buffer["rewards"].append(copy.deepcopy(rewards))
        dones = np.array(data['data/dones'])[:-1]
        dones[-1] = True
        per_obs_buffer["dones"].append(copy.deepcopy(dones))
        low_dim_obs_shape = 0

        for obs_name in obs_keys:

            value = np.array(data[f"data/{obs_name}"])

            per_obs_buffer[obs_name].append(value)
            if "rgb" not in obs_name:
                low_dim_obs_shape += value.shape[-1]
            else:
                use_image = True
        for key, val in per_obs_buffer.items():
            obs_buffer[key].extend(val)
        demo_count += 1
        if demo_count >= num_demos:
            break
    print(f"Loaded {demo_count} demos from {demo_path}")

    action_data = np.concatenate(obs_buffer["actions"], axis=0)
    action_data = (action_data - joint_limits[:, 0]) / (
        joint_limits[:, 1] - joint_limits[:, 0]) * 2 - 1

    obs_buffer.pop("actions")
    rewards_data = np.concatenate(obs_buffer["rewards"], axis=0)
    obs_buffer.pop("rewards")
    dones_data = np.concatenate(obs_buffer["dones"], axis=0)
    obs_buffer.pop("dones")

    last_lowdim_obs = []
    next_lowdim_obs = []
    for obs_name in obs_keys:
        if "rgb" not in obs_name:
            last_lowdim_obs.append(np.array(obs_buffer[obs_name])[:, :-1])
            next_lowdim_obs.append(np.array(obs_buffer[obs_name])[:, 1:])
        else:
            visual_obs_buffer[obs_name].append(obs_buffer[obs_name][:-1])
            next_visual_obs_buffer[obs_name].append(obs_buffer[obs_name][1:])

    lowdim_obs_data = np.concatenate(np.concatenate(last_lowdim_obs, axis=-1),
                                     axis=0)
    lowdim_next_obs_data = np.concatenate(np.concatenate(next_lowdim_obs,
                                                         axis=-1),
                                          axis=0)

    rewards_data /= 30

    if use_image:
        image_keys = copy.deepcopy(list(visual_obs_buffer.keys()))
        for image_key in image_keys:

            visual_obs_buffer["rgb"] = np.concatenate(
                visual_obs_buffer.pop(image_key), axis=0).astype(np.uint8)
            next_visual_obs_buffer["rgb"] = np.concatenate(
                next_visual_obs_buffer.pop(image_key), axis=0).astype(np.uint8)

        return dict(observations={"state": lowdim_obs_data}
                    | visual_obs_buffer,
                    actions=action_data,
                    rewards=rewards_data,
                    next_observations={"state": lowdim_next_obs_data}
                    | next_visual_obs_buffer,
                    dones=dones_data), use_image, low_dim_obs_shape, len(
                        lowdim_obs_data)

    return dict(
        observations=lowdim_obs_data,
        actions=action_data,
        rewards=rewards_data,
        next_observations=lowdim_next_obs_data,
        dones=dones_data), use_image, low_dim_obs_shape, len(lowdim_obs_data)


def populate_data_store_from_zarr_to_sb3(replay_buffer=None,
                                         demo_path=None,
                                         obs_keys=[],
                                         num_demos=100):
    demo_file = os.listdir(demo_path)

    joint_limits = np.array(
        [[-0.03, 0.03], [-0.03, 0.03], [-0.03, 0.03], [-0.05, 0.05],
         [-0.05, 0.05], [-0.05, 0.05], [-0.314, 2.23], [-0.349, 2.094],
         [-0.314, 2.23], [-0.314, 2.23], [-1.047, 1.047],
         [-0.46999997, 2.4429998], [-1.047, 1.047], [-1.047, 1.047],
         [-0.5059999, 1.8849999], [-1.2, 1.8999999], [-0.5059999, 1.8849999],
         [-0.5059999, 1.8849999], [-0.366, 2.0419998], [-1.34, 1.8799999],
         [-0.366, 2.0419998], [-0.366, 2.0419998]], )

    for file in demo_file[:num_demos]:
        if not file.endswith(".zarr"):
            continue
        zarr_path = os.path.join(demo_path, file)
        data = zarr.open(zarr_path, mode='r')

        actions = np.array(data['data/actions'])[:-1]

        rewards = np.array(data['data/rewards'])[:-1]

        dones = np.array(data['data/dones'])[:-1]
        dones[-1] = True

        lowdim_obs_shape = 0
        lowdim_obs = []

        for obs_name in obs_keys:

            value = np.array(data[f"data/{obs_name}"])

            if "rgb" not in obs_name:
                lowdim_obs_shape += value.shape[-1]
            else:
                use_image = True
            lowdim_obs.append(value)

        lowdim_obs = np.concatenate(lowdim_obs, axis=-1)

        if replay_buffer is not None:
            dones = dones.astype(np.float32)

            for i in range(len(actions)):

                replay_buffer.add(
                    obs=lowdim_obs[i],
                    next_obs=lowdim_obs[i + 1],
                    action=actions[i],
                    reward=rewards[i],
                    done=dones[i].astype(np.float32),
                    infos=[{}],
                )
            print(replay_buffer.pos)

    import pdb
    pdb.set_trace()
    return


if __name__ == "__main__":
    obs_keys = [
        'right_hand_joint_pos', "right_manipulated_object_pose",
        "right_target_object_pose"
    ]
    demo_path = "logs/data_0705/retarget_visionpro_data/rl_data/data/ours/image/bunny"
    # demo_data, use_image, low_dim_obs_shape, capacity = populate_data_store_from_zarr(
    #     demo_path, obs_keys, num_demos=20)
    demo_data, use_image, low_dim_obs_shape, capacity = populate_data_store_from_zarr_to_sb3(
        None, demo_path, obs_keys, num_demos=20)
