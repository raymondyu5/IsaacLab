import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scripts.workflows.hand_manipulation.env.rl_env.torch_layers import PointNetStateExtractor, ImageStateExtractor
import torch.nn as nn

visual_observation_space = spaces.Dict({
    "rgb":
    spaces.Box(
        low=0,
        high=255,
        shape=(1, 224, 224, 3),
        dtype=np.uint8,
    ),
    "state":
    spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(46, ),
        dtype=np.float32,
    ),
    "seg_pc":
    spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(10240, 3),
        dtype=np.float32,
    ),
})

pcd_observation_space = spaces.Dict({
    "seg_pc":
    spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(10240, 3),
        dtype=np.float32,
    ),
    "state":
    spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(46, ),
        dtype=np.float32,
    ),
})

lowdim_observation_space = spaces.Box(
    low=-np.inf,
    high=np.inf,
    shape=(37, ),
    dtype=np.float32,
)

action_space = spaces.Box(
    low=-1.0,
    high=1.0,
    shape=(22, ),
    dtype=np.float32,
)

latent_action_space = spaces.Box(
    low=-1.0,
    high=1.0,
    shape=(44, ),
    dtype=np.float32,
)

latent_action_space = spaces.Box(
    low=-1.0,
    high=1.0,
    shape=(44, ),
    dtype=np.float32,
)

only_latent_action_space = spaces.Box(
    low=-1.0,
    high=1.0,
    shape=(22, ),
    dtype=np.float32,
)
td3_cfg = {
    "seed": 42,
    # epoch * n_steps * nenvs: 500×512*8*8
    "n_timesteps": 1000,
    "policy": "MultiInputPolicy",
    "buffer_size": 10000,
    "batch_size": 512,
    "learning_starts": 120,
    "gradient_steps": 10,
    "tau": 0.002,
    "gamma": 0.98,
    # "target_policy_noise": 0.1,
    # "target_noise_clip": 0.15,
    # "action_noise": {"std": 0.2, "mean": 0.0},
    "policy_delay": 2,
    # "stats_window_size": 10,
    "train_freq": 120
}

pcd_policy_kwargs = {
    "policy_kwargs": {
        "features_extractor_class": PointNetStateExtractor,
        "features_extractor_kwargs": {
            "pc_key": ["seg_pc"],
            "local_channels": (64, 128, 256),
            "global_channels": (256, ),
            "use_bn": False,
            "state_mlp_size": (64, 64),
        },
        "net_arch": {
            "qf": [256, 128, 64],
            "pi": [256, 128, 64],
        },
        "activation_fn": nn.ReLU,
    }
}

state_policy_kwargs = dict(policy_kwargs=dict(net_arch=[256, 128, 64]))

visual_policy_kwargs = {
    "policy_kwargs": {
        "features_extractor_class": ImageStateExtractor,
        "features_extractor_kwargs": {
            "image_key": "rgb",
            "image_features_dim": 256,
            "state_mlp_size": (64, 64),
            "state_mlp_size": (64, 64),
        },
        "net_arch": {
            "qf": [256, 128, 64],
            "pi": [256, 128, 64],
        },
        "activation_fn": nn.ReLU,
    }
}

pcd_dataset_cfg = {
    "data_path": "/home/weirdlab/Documents/droid/logs/delta",
    "load_list": ["spider"],
    "horizon": 1,
    "pad_before": 0,
    "pad_after": 0,
    "obs_key": ["joint_positions", "gripper_position", "base_action"],
    "noise_key": ["joint_positions"],
    "noise_scale": {
        "joint_positions": 0.05
    },
    "seed": 42,
    "val_ratio": 0.0,
    "num_demo": 5,
    "max_train_episodes": None,
    "downsample_points": 2048,
    "pcd_noise": 0.0,
    "noise_extrinsic": True,
    "noise_extrinsic_parameter": [0.06, 0.2],
    "camera_id": ["CL8384200N1"],
    "crop_region": [
        -0.20,
        -0.40,
        0.02,
        0.85,
        0.40,
        0.70,
    ],
}
