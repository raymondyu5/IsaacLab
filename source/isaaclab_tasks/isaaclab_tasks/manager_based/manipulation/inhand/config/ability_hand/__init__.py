# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents, joint_pos_env_cfg

##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="Isaac-Ability-YCB-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point":
        f"{joint_pos_env_cfg.AbilityYCBEnvCfg_PLAY}:config.yaml",
        "rsl_rl_cfg_entry_point":
        f"{agents.__name__}.rsl_rl_ppo_cfg:CabinetPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)
