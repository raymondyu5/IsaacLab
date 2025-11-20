# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Script to run an environment with zero action agent."""
"""Launch Isaac Sim Simulator first."""

import time
import sys

sys.path.append("submodule/submitit-tools")
from submitit_tools.configs import SubmititExecutorConfig, WandbConfig
from submitit_tools.jobs import BaseJob, SubmititState, grid_search_job_configs
from isaaclab.app import AppLauncher
from scripts.workflows.utils.parse_setting import save_params_to_yaml, parser

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Script to train RL agent with Stable Baselines3.

Since Stable-Baselines3 does not support buffers living on GPU directly,
we recommend using smaller number of environments. Otherwise,
there will be significant overhead in GPU->CPU transfer.
"""
"""Launch Isaac Sim Simulator first."""

import argparse
import gymnasium as gym
import numpy as np
import os
import random
from datetime import datetime

parser.add_argument(
    "--add_right_hand",
    action="store_true",
)
parser.add_argument(
    "--add_left_hand",
    action="store_true",
)
parser.add_argument(
    "--action_framework",
    default=None,
)

parser.add_argument(
    "--target_object_name",
    type=str,
    default=None,
)

import gymnasium as gym
import torch

import os

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BaseJobConfig:

    target_object_name: Optional[str] = None
    checkpoint_path: Optional[str] = "logs/checkpoints"

    def __post_init__(self):
        pass

    def __str__(self):
        contents = "\n".join(f"\t{key}: {value}"
                             for key, value in self.__dict__.items())
        return "{\n" + contents + "\n}"


def setup_env(args_cli, save_config):
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(args_cli.task,
                            device=args_cli.device,
                            num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric,
                            config_yaml=save_config)

    return gym.make(args_cli.task, cfg=env_cfg).unwrapped


class EnvConfig(BaseJobConfig):
    pass


def generate_job_configs(base_config,
                         object_list,
                         job_creation_fn=None,
                         job_cls=None):
    result_configs = []
    for object_name in object_list:
        config = base_config.copy()
        config["target_object_name"] = object_name

        job_config = EnvConfig(**config)

        if job_creation_fn is None:
            job = job_cls(**job_config)
        else:
            job = job_creation_fn(**vars(job_config))
        result_configs.append(job)
    return result_configs


def generate_train_configs():

    def creation_fn(checkpoint_path=None, target_object_name=None):
        job = EnvConfig(target_object_name=target_object_name,
                        checkpoint_path=checkpoint_path)
        return (job, None)

    config = {
        "checkpoint_path": "logs/checkpoints",
    }
    object_list = [
        "tomato_soup_can",
    ]
    configs = generate_job_configs(config,
                                   object_list,
                                   job_creation_fn=creation_fn)
    job_cfgs = [item[0] for item in configs]
    return job_cfgs


class EnvRunJob(BaseJob):

    def __init__(self, job_config, wandb_config=None):
        super().__init__(job_config, wandb_config)
        self.job_config = job_config
        assert WandbConfig is not None, "This Job uses Wandb"

    def _initialize(self):
        # launch omniverse app
        AppLauncher.add_app_launcher_args(parser)
        args_cli, hydra_args = parser.parse_known_args()
        app_launcher = AppLauncher(args_cli)
        self.simulation_app = app_launcher.app
        from scripts.workflows.hand_manipulation.env.rl_env.rl_wrapper import RLDatawrapperEnv
        """Rest everything follows."""

        args_cli.target_object_name = self.job_config.target_object_name

        save_config, config = save_params_to_yaml(args_cli,
                                                  args_cli.log_dir,
                                                  random_shuffle=False)

        object_name = args_cli.target_object_name
        save_config["params"]["multi_cluster_rigid"]["right_hand_object"][
            "objects_list"] = [object_name]

        # create environment
        save_config["params"]["add_right_hand"] = args_cli.add_right_hand
        save_config["params"]["add_left_hand"] = args_cli.add_left_hand
        save_config["params"]["num_envs"] = args_cli.num_envs
        save_config["params"]["eval_mode"] = True

        self.env = setup_env(args_cli, save_config)
        self.env.reset()

        args_cli.load_path = "raw_data/" + object_name

        self.rl_env = RLDatawrapperEnv(
            self.env,
            save_config,
            args_cli=args_cli,
            use_relative_pose=True if "Rel" in args_cli.task else False,
            replay_mode=True,
        )

    def _job_call(self):

        self.rl_env.replay()

        self.env.close()
        self.simulation_app.close()

    def _save_checkpoint(self):
        pass


def main():
    executor_config = SubmititExecutorConfig(root_folder="mnist_submitit_logs",
                                             slurm_partition="gpu-a40",
                                             slurm_name="submitit-test",
                                             timeout_min=60 * 2,
                                             cpus_per_task=6,
                                             mem_gb=100)
    job_configs = generate_train_configs()
    state = SubmititState(job_cls=EnvRunJob,
                          executor_config=executor_config,
                          job_run_configs=job_configs,
                          job_wandb_configs=None,
                          with_progress_bar=True,
                          max_retries=4,
                          num_concurrent_jobs=8)

    state.run_all_jobs()

    for result in state.results:
        print(result)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
