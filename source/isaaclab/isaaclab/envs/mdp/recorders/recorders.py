# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import torch
from collections.abc import Sequence

from isaaclab.managers.recorder_manager import RecorderTerm


class InitialStateRecorder(RecorderTerm):
    """Recorder term that records the initial state of the environment after reset."""

    def record_post_reset(self, env_ids: Sequence[int] | None):
        def extract_env_ids_values(value):
            nonlocal env_ids
            if isinstance(value, dict):
                return {k: extract_env_ids_values(v) for k, v in value.items()}
            return value[env_ids]

        return "initial_state", extract_env_ids_values(self._env.scene.get_state(is_relative=True))


class PostStepStatesRecorder(RecorderTerm):
    """Recorder term that records the state of the environment at the end of each step."""

    def record_post_step(self):
        return "states", self._env.scene.get_state(is_relative=True)


class PreStepActionsRecorder(RecorderTerm):
    """Recorder term that records the actions in the beginning of each step."""

    def record_pre_step(self):
        return "actions", self._env.action_manager.action


class PreStepFlatPolicyObservationsRecorder(RecorderTerm):
    """Recorder term that records policy observations and instantaneous states.

    Saves both:
    - obs: Full policy observations (for action translator operating in observation space)
    - state: Instantaneous positional state (for ID models with temporal sequences)

    State representation: s_t = [q_t, k_obj_t]
    - q_t: Robot joint positions
    - k_obj_t: Object pose (position + quaternion)

    Dimensions are automatically determined from the environment.
    Models can infer velocities from sequences [s_{t-1}, s_t, s_{t+1}, s_{t+2}].
    """

    def record_pre_step(self):
        """Record both policy observations and instantaneous positional states."""
        # Get full policy observations (needed for action translator)
        obs_full = self._env.obs_buf["policy"]

        # Extract instantaneous positional state from scene
        scene_state = self._env.scene.get_state(is_relative=True)

        # Extract robot joint positions
        robot_joint_pos = scene_state['articulation']['robot']['joint_position']  # (num_envs, joint_dim)

        # Extract object pose (position + quaternion)
        obj_pose = scene_state['rigid_object']['object']['root_pose']  # (num_envs, 7)

        # Construct instantaneous state: [q_t, k_obj_t]
        state = torch.cat([
            robot_joint_pos,  # Current joint positions
            obj_pose          # Current object pose
        ], dim=-1)

        # Return as nested dictionary structure
        return "policy_data", {"obs": obs_full, "state": state}


class PostStepProcessedActionsRecorder(RecorderTerm):
    """Recorder term that records processed actions at the end of each step."""

    def record_post_step(self):
        processed_actions = None

        # Loop through active terms and concatenate their processed actions
        for term_name in self._env.action_manager.active_terms:
            term_actions = self._env.action_manager.get_term(term_name).processed_actions.clone()
            if processed_actions is None:
                processed_actions = term_actions
            else:
                processed_actions = torch.cat([processed_actions, term_actions], dim=-1)

        return "processed_actions", processed_actions
