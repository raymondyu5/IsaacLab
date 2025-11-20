# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym
import os

from . import agents, ik_abs_env_cfg, ik_rel_env_cfg, joint_pos_env_cfg

##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="Isaac-Lift-DeformCube-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point":
        f"{joint_pos_env_cfg.FrankaDeformCubeLiftEnvCfg}:test.yaml",
        # joint_pos_env_cfg.FrankaDeformCubeLiftEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.LiftDeformCubePPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-DeformCube-Franka-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point":
        joint_pos_env_cfg.FrankaDeformCubeLiftEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.LiftDeformCubePPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Absolute Pose Control
##

gym.register(
    id="Isaac-Lift-DeformCube-Franka-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point":
        f"{ik_abs_env_cfg.FrankaDeformCubeLiftEnvCfg}:test.yaml",
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="Isaac-Lift-DeformCube-Franka-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point":
        f"{ik_rel_env_cfg.FrankaDeformCubeLiftEnvCfg}:test.yaml",
        "robomimic_bc_cfg_entry_point":
        os.path.join(agents.__path__[0], "robomimic/bc.json"),
    },
    disable_env_checker=True,
)
