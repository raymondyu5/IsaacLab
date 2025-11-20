# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import mdp
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CurriculumTermCfg as CurrTerm

import numpy as np


def initial_final_interpolate_fn(env: ManagerBasedRLEnv, env_id, data,
                                 initial_value, final_value,
                                 difficulty_term_str):
    """
    Interpolate between initial value iv and final value fv, for any arbitrarily
    nested structure of lists/tuples in 'data'. Scalars (int/float) are handled
    at the leaves.
    """
    # get the fraction scalar on the device
    difficulty_term: DifficultyScheduler = getattr(env.curriculum_manager.cfg,
                                                   difficulty_term_str).func
    frac = difficulty_term.difficulty_frac

    # if frac < 0.1:  # the warm-up phase need to be larger than 80% sucess rate
    #     # no-op during start, since the difficulty fraction near 0 is wasting of resource.
    #     return mdp.modify_env_param.NO_CHANGE

    # convert iv/fv to tensors, but we'll peel them apart in recursion
    initial_value_tensor = torch.tensor(initial_value, device=env.device)
    final_value_tensor = torch.tensor(final_value, device=env.device)
    factor = _recurse(initial_value_tensor.tolist(),
                      final_value_tensor.tolist(), data, frac)

    return factor


def _recurse(iv_elem, fv_elem, data_elem, frac):
    # If it's a sequence, rebuild the same type with each element recursed
    if isinstance(data_elem,
                  Sequence) and not isinstance(data_elem, (str, bytes)):
        # Note: we assume initial value element and final value element have the same structure as data
        return type(data_elem)(
            _recurse(iv_e, fv_e, d_e, frac)
            for iv_e, fv_e, d_e in zip(iv_elem, fv_elem, data_elem))

    # Otherwise it's a leaf scalar: do the interpolation
    if isinstance(fv_elem, list):
        fv_elem = torch.tensor(fv_elem).to(torch.float32).to(frac.device)
        iv_elem = torch.tensor(iv_elem).to(torch.float32).to(frac.device)

    new_val = frac * (fv_elem - iv_elem) + iv_elem

    if isinstance(data_elem, int):
        return int(new_val.item())
    else:
        # cast floats or any numeric
        return new_val


class DifficultyScheduler(ManagerTermBase):
    """Adaptive difficulty scheduler for curriculum learning.

    Tracks per-environment difficulty levels and adjusts them based on task performance. Difficulty increases when
    position/orientation errors fall below given tolerances, and decreases otherwise (unless `promotion_only` is set).
    The normalized average difficulty across environments is exposed as `difficulty_frac` for use in curriculum
    interpolation.

    Args:
        cfg: Configuration object specifying scheduler parameters.
        env: The manager-based RL environment.

    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        init_difficulty = self.cfg.params.get("init_difficulty", 0)
        max_difficulty = self.cfg.params.get("max_difficulty", 50)
        self.current_adr_difficulties = torch.ones(
            env.num_envs, device=env.device) * init_difficulty
        self.difficulty_frac = torch.mean(self.current_adr_difficulties) / max(
            max_difficulty, 1)

    def get_state(self):
        return self.current_adr_difficulties

    def set_state(self, state: torch.Tensor):
        self.current_adr_difficulties = state.clone().to(self._env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        init_difficulty: int = 0,
        min_difficulty: int = 0,
        max_difficulty: int = 50,
        promotion_only: bool = False,
        successs_threshold: float = 0.8,
        suuccess_block: float = 0.6,
    ):

        asset: Articulation = env.scene[asset_cfg.name]
        object: RigidObject = env.scene[object_cfg.name]
        object_pose = object._data.root_state_w[:, :3]
        object_pose[:, :3] -= env.scene.env_origins

        move_up = object_pose[:, 2] > 0.3
        # demot = self.current_adr_difficulties[
        #     env_ids] if promotion_only else self.current_adr_difficulties[
        #         env_ids] - 1
        # self.current_adr_difficulties[env_ids] = torch.where(
        #     move_up,
        #     self.current_adr_difficulties[env_ids] + 1,
        #     demot,
        # ).clamp(min=min_difficulty, max=max_difficulty)
        # self.difficulty_frac = torch.mean(self.current_adr_difficulties) / max(
        #     max_difficulty, 1)

        success_rate = move_up.sum().item() / env.num_envs
        if success_rate >= successs_threshold:

            self.current_adr_difficulties[:] += 1

        else:
            if not promotion_only and suuccess_block > success_rate:
                self.current_adr_difficulties[:] -= 1
                self.current_adr_difficulties.clamp_(min=min_difficulty,
                                                     max=max_difficulty)
        self.difficulty_frac = torch.mean(self.current_adr_difficulties) / max(
            max_difficulty, 1)

        print(f"[INFO] Difficulty levels: {self.difficulty_frac}",
              "success rate:",
              move_up.sum().item() / env.num_envs)

        return self.difficulty_frac


def config_adr(obj, env_cfg):

    if env_cfg["params"]["add_right_hand"]:
        handness = "right"
    else:
        handness = "left"

    # adr stands for automatic/adaptive domain randomization
    adr = CurrTerm(func=DifficultyScheduler,
                   params={
                       "asset_cfg":
                       SceneEntityCfg(f"{handness}_hand"),
                       "object_cfg":
                       SceneEntityCfg(f"{handness}_hand_object"),
                       "init_difficulty":
                       env_cfg["params"]["adr"]["init_difficulty"],
                       "min_difficulty":
                       env_cfg["params"]["adr"]["min_difficulty"],
                       "max_difficulty":
                       env_cfg["params"]["adr"]["max_difficulty"],
                       "promotion_only":
                       env_cfg["params"]["adr"].get("promotion_only", False),
                       "successs_threshold":
                       env_cfg["params"]["adr"].get("successs_threshold", 0.8),
                       "suuccess_block":
                       env_cfg["params"]["adr"].get("suuccess_block", 0.6),
                   })
    setattr(obj.curriculum, "adr", adr)
    init_factor = env_cfg["params"]["adr"]["init_difficulty"] / env_cfg[
        "params"]["adr"]["max_difficulty"]

    object_region_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address":
            f"events.reset_{handness}_hand_object.params.pose_range",
            "modify_fn": initial_final_interpolate_fn,
            "modify_params": {
                "initial_value":
                np.array(
                    getattr(obj.events, f"reset_{handness}_hand_object").
                    params["default_pose_range"]) * 0.0,
                "final_value":
                np.array(
                    getattr(obj.events, f"reset_{handness}_hand_object").
                    params["default_pose_range"]),
                "difficulty_term_str":
                "adr"
            },
        },
    )

    reset_hand_object_fun = getattr(obj.events,
                                    f"reset_{handness}_hand_object")
    reset_hand_object_fun.params["pose_range"] *= init_factor
    setattr(obj.events, f"reset_{handness}_hand_object", reset_hand_object_fun)
    setattr(obj.curriculum, "object_region_adr", object_region_adr)
