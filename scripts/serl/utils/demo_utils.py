import zarr

from collections import defaultdict

import copy

import numpy as np
import os
# ['right_contact_obs', 'right_ee_pose', 'right_hand_joint_pos', 'right_manipulated_object_pose', 'right_object_in_tip', 'right_target_object_pose']
from typing import Optional

import gymnasium as gym
from typing import Union, Iterable
from threading import Lock
from typing import List, Optional, TypeVar
from scripts.serl.utils.memory_utils import ReplayBufferDataStore

import yaml
import matplotlib.pyplot as plt
from scripts.serl.utils.data_buffer import MemoryEfficientReplayBufferDataStore


def populate_data_store_from_zarr(demo_path,
                                  obs_keys,
                                  use_diffusion_model=False,
                                  num_demos=100):
    demo_file = os.listdir(demo_path)
    obs_buffer = defaultdict(list)
    obs_buffer["last_action"] = []
    demo_count = 0
    use_image = False

    for file in demo_file:
        if not file.endswith(".zarr"):
            continue
        zarr_path = os.path.join(demo_path, file)
        data = zarr.open(zarr_path, mode='r')
        per_obs_buffer = defaultdict(list)
        actions = np.array(data['data/actions'])

        if use_diffusion_model:

            obs_buffer["last_action"].append(actions[..., -16:])

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
    obs_buffer.pop("actions")
    rewards_data = np.concatenate(obs_buffer["rewards"], axis=0)
    obs_buffer.pop("rewards")
    dones_data = np.concatenate(obs_buffer["dones"], axis=0)
    obs_buffer.pop("dones")

    lowdim_obs_data = []
    lowdim_next_obs_data = []
    visual_obs_buffer = defaultdict(list)
    next_visual_obs_buffer = defaultdict(list)
    if use_diffusion_model:
        obs_keys = obs_keys + ["last_action"]

    for key in obs_keys:
        if "rgb" in key:
            for i in range(len(obs_buffer[key])):
                visual_obs_buffer[key].append(obs_buffer[key][i][:-1])
                next_visual_obs_buffer[key].append(obs_buffer[key][i][1:])

        else:
            for i in range(len(obs_buffer[key])):

                lowdim_obs_data.append(obs_buffer[key][i][:-1])
                lowdim_next_obs_data.append(obs_buffer[key][i][1:])

    lowdim_obs_data = np.concatenate(lowdim_obs_data, axis=0)
    lowdim_next_obs_data = np.concatenate(lowdim_next_obs_data, axis=0)

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


def normalize_action(action, action_range):
    joint_limits = np.stack(
        [np.min(action[:, 6:], axis=0),
         np.max(action[:, 6:], axis=0)],
        axis=-1)

    arm_action_range = np.stack([
        np.array([-action_range[0]] * 3 + [-action_range[1]] * 3),
        np.array([action_range[0]] * 3 + [action_range[1]] * 3)
    ],
                                axis=-1) / 20
    action_bounds = np.concatenate([arm_action_range, joint_limits], axis=0)

    # assume `action` is already loaded, shape (N, D)
    sub_action = action[:, 6:]  # shape (N, 16)

    num_dims = sub_action.shape[1]
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))  # 16 dims in 4x4 grid

    for i in range(num_dims):
        ax = axes[i // 4, i % 4]
        ax.hist(sub_action[:, i], bins=50, color="steelblue", alpha=0.7)
        ax.set_title(
            f"Dim {i+6}")  # +6 because we started slicing from index 6
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    #
    plt.savefig(os.path.join("logs/", "action_distribution.png"))

    normazlied_action = (action - action_bounds[:, 0]) / (
        action_bounds[:, 1] - action_bounds[:, 0]) * 2.0 - 1.0

    normazlied_action = np.clip(normazlied_action, -1.0, 1.0)

    return normazlied_action, action_bounds


def constuct_buffer(low_dim_obs_shape, use_image, capacity, obs,
                    normazlied_action, rewards, next_obs, dones):
    from scripts.serl.env.serl_wrapper import ImageCustomSerlEnvWrapper, StateCustomSerlEnvWrapper

    if use_image:
        obs_space, action_space = ImageCustomSerlEnvWrapper(
        ).observation_space, ImageCustomSerlEnvWrapper().action_space

        replay_buffer = MemoryEfficientReplayBufferDataStore(
            obs_space,
            action_space,
            capacity=capacity,
            image_keys=["rgb"],
        )

    else:
        action_space = StateCustomSerlEnvWrapper().action_space
        obs_space = gym.spaces.Box(
            low=-np.inf,  # or np.full(52, -1.0)
            high=np.inf,  # or np.full(52,  1.0)
            shape=(low_dim_obs_shape, ),
            dtype=np.float32)
        replay_buffer = ReplayBufferDataStore(
            obs_space,
            action_space,
            capacity=capacity,
        )
    for t in range(len(rewards)):
        obs_t = {
            k: v[t]
            for k, v in obs.items()
        }  # dict with 'state' and 'rgb' at step t
        next_obs_t = {k: v[t] for k, v in next_obs.items()}

        data_dict = {
            "observations": obs_t,
            "actions": normazlied_action[t],
            "rewards": rewards[t],
            "next_observations": next_obs_t,
            "dones": dones[t],
            "masks": 1 - dones[t],
        }
        replay_buffer.insert(data_dict)
    return replay_buffer


def make_zarr_replay_buffer(
    demo_path,
    obs_keys: Optional[List[str]] = None,
    capacity: int = 1_000_000,
    action_range: Union[float, Iterable[float]] = 1.0,
    use_diffusion_model: bool = False,
    num_demos: int = 100,
):

    demo_data, use_image, low_dim_obs_shape, capacity = populate_data_store_from_zarr(
        demo_path,
        obs_keys,
        use_diffusion_model=use_diffusion_model,
        num_demos=num_demos)

    obs = demo_data["observations"]
    action = demo_data["actions"]
    rewards = demo_data["rewards"]
    next_obs = demo_data["next_observations"]
    dones = demo_data["dones"]
    normazlied_action, action_bounds = normalize_action(action, action_range)

    replay_buffer = constuct_buffer(low_dim_obs_shape, use_image, capacity,
                                    obs, normazlied_action, rewards, next_obs,
                                    dones)

    return replay_buffer, normazlied_action, action_bounds, dones


if __name__ == "__main__":
    obs_keys = ['right_hand_joint_pos', "rgb_0"]
    demo_path = "logs/data_0705/retarget_visionpro_data/rl_data/plush/demo_image/bunny"
    with open("source/config/task/hand_env/leap_franka/rl_env_plush_cam.yml",
              'r') as file:
        env_config = yaml.safe_load(file)

    action_range = env_config["params"]["Task"]["action_range"]

    replay_buffer = make_zarr_replay_buffer(demo_path,
                                            obs_keys,
                                            action_range=action_range,
                                            num_demos=2)[0]
