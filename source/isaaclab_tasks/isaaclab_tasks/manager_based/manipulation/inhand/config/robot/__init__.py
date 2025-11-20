# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

##
# Joint Position Control
##
from .grasp import grasp_ik_abs_env_cfg, grasp_ik_rel_env_cfg, grasp_joint_pos_env_cfg, grasp_joint_rel_env_cfg

gym.register(
    id="Isaac-Hand-Robot-YCB-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point":
        f"{grasp_joint_pos_env_cfg.HandYCBEnvCfg_PLAY}:config.yaml",
        "rsl_rl_cfg_entry_point":
        f"{agents.__name__}.rsl_rl_ppo_cfg:CabinetPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Hand-Robot-YCB-Joint-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point":
        f"{grasp_joint_rel_env_cfg.HandYCBEnvCfg}:config.yaml",
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Absolute Pose Control
##

gym.register(
    id="Isaac-Hand-Robot-YCB-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point":
        f"{grasp_ik_abs_env_cfg.HandYCBEnvCfg}:config.yaml",
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="Isaac-Hand-Robot-YCB-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point":
        f"{grasp_ik_rel_env_cfg.HandYCBEnvCfg}:config.yaml",
    },
    disable_env_checker=True,
)
from .place import place_ik_abs_env_cfg, place_ik_rel_env_cfg, place_joint_pos_env_cfg, place_joint_rel_env_cfg

gym.register(
    id="Isaac-Hand-Robot-Place-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point":
        f"{place_joint_pos_env_cfg.HandYCBEnvCfg_PLAY}:config.yaml",
        "rsl_rl_cfg_entry_point":
        f"{agents.__name__}.rsl_rl_ppo_cfg:CabinetPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Hand-Robot-Place-Joint-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point":
        f"{place_joint_rel_env_cfg.HandYCBEnvCfg}:config.yaml",
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Absolute Pose Control
##

gym.register(
    id="Isaac-Hand-Robot-Place-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point":
        f"{place_ik_abs_env_cfg.HandYCBEnvCfg}:config.yaml",
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="Isaac-Hand-Robot-Place-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point":
        f"{place_ik_rel_env_cfg.HandYCBEnvCfg}:config.yaml",
    },
    disable_env_checker=True,
)

from .open import open_ik_abs_env_cfg, open_ik_rel_env_cfg, open_joint_pos_env_cfg, open_joint_rel_env_cfg

gym.register(
    id="Isaac-Hand-Robot-Open-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point":
        f"{open_joint_pos_env_cfg.HandYCBEnvCfg_PLAY}:config.yaml",
        "rsl_rl_cfg_entry_point":
        f"{agents.__name__}.rsl_rl_ppo_cfg:CabinetPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Hand-Robot-Open-Joint-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point":
        f"{open_joint_rel_env_cfg.HandYCBEnvCfg}:config.yaml",
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Absolute Pose Control
##

gym.register(
    id="Isaac-Hand-Robot-Open-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point":
        f"{open_ik_abs_env_cfg.HandYCBEnvCfg}:config.yaml",
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="Isaac-Hand-Robot-Open-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point":
        f"{open_ik_rel_env_cfg.HandYCBEnvCfg}:config.yaml",
    },
    disable_env_checker=True,
)

from .ungraspable import ungraspable_ik_abs_env_cfg, ungraspable_ik_rel_env_cfg, ungraspable_joint_pos_env_cfg, ungraspable_joint_rel_env_cfg

gym.register(
    id="Isaac-Hand-Robot-Ungraspable-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point":
        f"{ungraspable_joint_pos_env_cfg.HandYCBEnvCfg_PLAY}:config.yaml",
        "rsl_rl_cfg_entry_point":
        f"{agents.__name__}.rsl_rl_ppo_cfg:CabinetPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Hand-Robot-Ungraspable-Joint-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point":
        f"{ungraspable_joint_rel_env_cfg.HandYCBEnvCfg}:config.yaml",
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Absolute Pose Control
##

gym.register(
    id="Isaac-Hand-Robot-Ungraspable-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point":
        f"{ungraspable_ik_abs_env_cfg.HandYCBEnvCfg}:config.yaml",
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="Isaac-Hand-Robot-Ungraspable-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point":
        f"{ungraspable_ik_rel_env_cfg.HandYCBEnvCfg}:config.yaml",
    },
    disable_env_checker=True,
)

from .stack import stack_ik_abs_env_cfg, stack_ik_rel_env_cfg, stack_joint_pos_env_cfg, stack_joint_rel_env_cfg

gym.register(
    id="Isaac-Hand-Robot-Stack-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point":
        f"{stack_joint_pos_env_cfg.HandYCBEnvCfg_PLAY}:config.yaml",
        "rsl_rl_cfg_entry_point":
        f"{agents.__name__}.rsl_rl_ppo_cfg:CabinetPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Hand-Robot-Stack-Joint-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point":
        f"{stack_joint_rel_env_cfg.HandYCBEnvCfg}:config.yaml",
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Absolute Pose Control
##

gym.register(
    id="Isaac-Hand-Robot-Stack-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point":
        f"{stack_ik_abs_env_cfg.HandYCBEnvCfg}:config.yaml",
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="Isaac-Hand-Robot-Stack-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point":
        f"{stack_ik_rel_env_cfg.HandYCBEnvCfg}:config.yaml",
    },
    disable_env_checker=True,
)
