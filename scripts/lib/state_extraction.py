"""
134D state extraction utilities. (make comments universal later, not just 134 but currently)

Provides consistent state extraction for both:
1. Live environment observations (torch tensors) during evaluation
2. HDF5 dataset files (numpy arrays) during training

State representation: s_t = [q_t, q̇_t, q_obj,t, hand_tips_t, f_contact,t, k_target,t]
- q_t: Robot joint positions in configuration space (23 dims, rad or m)
- q̇_t: Robot joint velocities in configuration space (23 dims, rad/s or m/s)
- q_obj,t: Object orientation in robot base frame (4 dims: quaternion)
- hand_tips_t: Full hand state in robot base frame (65 dims: 5 bodies × 13D state)
  - Each body: pos(3) + quat(4) + lin_vel(3) + ang_vel(3)
  - Bodies: palm, index, middle, ring, thumb
- f_contact,t: Contact forces in robot base frame (12 dims: 4 tips × 3D force)
- k_target,t: Target object pose in robot base frame (7 dims: 3 pos + 4 quat)

Total: 134 dims

Note: Object position and velocity are NOT included (RL policy trained without them).
Only object orientation (quaternion) is included, matching what the RL policy used.

Frame conventions:
- Joint positions/velocities: Configuration space (frame-independent scalars for each DOF)
- All spatial observations: Robot base frame (not world frame)
"""

import torch
import numpy as np
import h5py
from typing import Tuple, Union


class StateExtractor:
    """
    Extract 134D Markovian states from either live environments or HDF5 datasets.

    This class provides a unified interface for extracting the same 134D state
    representation regardless of the data source (live simulation or offline dataset).
    """

    def __init__(self):
        pass

    def extract_from_env(self, env, obs_buf=None) -> torch.Tensor:
        """
        Extract 134D Markovian state from live IsaacLab environment.

        Used during policy evaluation to extract the reduced state representation
        from the full environment state.

        Args:
            env: IsaacLab environment with scene containing robot and object
            obs_buf: Optional observation buffer. If None, will use env.obs_buf["policy"]

        Returns:
            state_134d: (num_envs, 134) tensor containing the Markovian state
        """
        scene = env.scene

        # Robot state (46D total: 23D pos + 23D vel)
        # Note: Joint positions and velocities are in configuration space (joint angles/positions in rad/m)
        # They are frame-independent scalars representing each joint's DOF state, not spatial coordinates.
        robot_pos = scene.articulations["robot"].data.joint_pos  # (num_envs, 23)
        robot_vel = scene.articulations["robot"].data.joint_vel  # (num_envs, 23)

        # Get full policy observation if not provided
        if obs_buf is None:
            obs_buf = env.obs_buf["policy"]  # (num_envs, policy_obs_dim)

        # Calculate per-timestep dimension from PolicyCfg:
        # object_quat_b(4) + target_pose_b(7) + actions(22) = 33D
        # With history=5: 33 * 5 = 165D total
        per_timestep_dim = 33
        history_length = 5

        # Extract from the LATEST timestep (last per_timestep_dim values)
        # Latest timestep starts at index: (history_length - 1) * per_timestep_dim
        latest_start_idx = (history_length - 1) * per_timestep_dim

        # Object orientation in robot base frame (4D quaternion)
        # Extract from latest timestep in policy observation buffer
        obj_quat = obs_buf[:, latest_start_idx + 0:latest_start_idx + 4]  # object_quat_b (4D)

        # Target object pose (7D) - from latest policy observation
        target_pose = obs_buf[:, latest_start_idx + 4:latest_start_idx + 11]  # (num_envs, 7)

        # Get proprio observation buffer for fingertips and contact forces
        proprio_obs = env.obs_buf["proprio"]  # (num_envs, proprio_obs_dim)

        # Calculate dimensions from ProprioObsCfg:
        # joint_pos(23) + joint_vel(23) + hand_tips_state_b(5 bodies * 13) + contact(12)
        # Total per timestep: 23 + 23 + 65 + 12 = 123D
        # With history=5: 123 * 5 = 615D total
        proprio_per_timestep = 123
        proprio_latest_start = (history_length - 1) * proprio_per_timestep

        # Hand tips full state (65D) - extract all 5 bodies × 13D state
        # hand_tips_state_b has 5 bodies (palm + 4 fingertips), each with 13D state
        # Each 13D = pos(3) + quat(4) + lin_vel(3) + ang_vel(3)
        # Hand tips start at joint_pos(23) + joint_vel(23) = 46
        hand_tips_start = 46
        hand_tips_full = proprio_obs[:, proprio_latest_start + hand_tips_start:proprio_latest_start + hand_tips_start + 65]  # (num_envs, 65)

        # Contact forces (12D) - last 12 dims of each timestep
        contact_forces = proprio_obs[:, proprio_latest_start + 111:proprio_latest_start + 123]  # (num_envs, 12)

        # Concatenate all components
        state_134d = torch.cat([
            robot_pos,        # 23D
            robot_vel,        # 23D
            obj_quat,         # 4D (orientation only)
            hand_tips_full,   # 65D (full hand state)
            contact_forces,   # 12D
            target_pose,      # 7D
        ], dim=-1)  # (num_envs, 134)

        assert state_134d.shape[-1] == 134, f"Expected 134D state, got {state_134d.shape[-1]}D"

        return state_134d

    def extract_from_dataset(self, demo_group: h5py.Group) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract 134D Markovian states from HDF5 dataset file.

        Used during dataset loading for training ID/BC models on the reduced state representation.

        Expected HDF5 structure:
        - demo['states']['articulation']['robot']['joint_position']: (T, 23)
        - demo['states']['articulation']['robot']['joint_velocity']: (T, 23)
        - demo['obs']: (T, 165) - policy observations with history (165D)
        - demo['proprio_obs']: (T, 615) - proprio observations with history (615D)
        - demo['actions']: (T, action_dim)

        Args:
            demo_group: HDF5 demo group with states format

        Returns:
            states: (T, 134) numpy array containing the Markovian state sequence
            actions: (T, action_dim) numpy array of aligned actions
        """
        # Validate required structure
        if 'states' not in demo_group:
            raise KeyError(
                f"Expected 'states' in demo group. "
                f"Found keys: {list(demo_group.keys())}"
            )

        states_group = demo_group['states']

        # Extract robot data (46D total)
        robot_pos = np.array(states_group['articulation']['robot']['joint_position'])  # (T, 23)
        robot_vel = np.array(states_group['articulation']['robot']['joint_velocity'])  # (T, 23)

        # Extract policy and proprio observations
        if 'obs' not in demo_group:
            raise KeyError(
                f"Expected 'obs' in demo group. "
                f"Found keys: {list(demo_group.keys())}"
            )

        if 'proprio_obs' not in demo_group:
            raise KeyError(
                f"Expected 'proprio_obs' in demo group. "
                f"Found keys: {list(demo_group.keys())}"
            )

        policy_obs = np.array(demo_group['obs'])  # (T, 165)
        proprio_obs = np.array(demo_group['proprio_obs'])  # (T, 615)

        # Calculate per-timestep dimension from PolicyCfg:
        # object_quat_b(4) + target_pose_b(7) + actions(22) = 33D
        # With history=5: 33 * 5 = 165D total
        per_timestep_dim = 33
        history_length = 5

        # Extract from the LATEST timestep (last per_timestep_dim values)
        latest_start_idx = (history_length - 1) * per_timestep_dim

        # Object orientation in robot base frame (4D quaternion)
        # Extract from latest timestep in policy observation
        obj_quat = policy_obs[:, latest_start_idx + 0:latest_start_idx + 4]  # object_quat_b (4D)

        # Target object pose (7D) - from latest policy observation
        target_pose = policy_obs[:, latest_start_idx + 4:latest_start_idx + 11]  # (T, 7)

        # Calculate dimensions from ProprioObsCfg:
        # joint_pos(23) + joint_vel(23) + hand_tips_state_b(5 bodies * 13) + contact(12)
        # Total per timestep: 23 + 23 + 65 + 12 = 123D
        # With history=5: 123 * 5 = 615D total
        proprio_per_timestep = 123
        proprio_latest_start = (history_length - 1) * proprio_per_timestep

        # Hand tips full state (65D) - extract all 5 bodies × 13D state
        # hand_tips_state_b has 5 bodies (palm + 4 fingertips), each with 13D state
        # Each 13D = pos(3) + quat(4) + lin_vel(3) + ang_vel(3)
        # Hand tips start at joint_pos(23) + joint_vel(23) = 46
        hand_tips_start = 46
        hand_tips_full = proprio_obs[:, proprio_latest_start + hand_tips_start:proprio_latest_start + hand_tips_start + 65]  # (T, 65)

        # Contact forces (12D) - last 12 dims of each timestep
        contact_forces = proprio_obs[:, proprio_latest_start + 111:proprio_latest_start + 123]  # (T, 12)

        # Load actions
        if 'actions' not in demo_group:
            raise KeyError(
                f"Expected 'actions' in demo group. "
                f"Found keys: {list(demo_group.keys())}"
            )
        actions = np.array(demo_group['actions'])

        # Concatenate all components
        states = np.concatenate([
            robot_pos,        # 23 dims
            robot_vel,        # 23 dims
            obj_quat,         # 4 dims (orientation only)
            hand_tips_full,   # 65 dims (full hand state)
            contact_forces,   # 12 dims
            target_pose       # 7 dims
        ], axis=1)  # Total: 134 dims

        # Validate dimensions
        assert states.shape[0] == actions.shape[0], \
            f"States and actions must have same length. " \
            f"Got states: {states.shape[0]}, actions: {actions.shape[0]}"

        assert states.shape[1] == 134, \
            f"Expected state dimension 134, got {states.shape[1]}"

        return states, actions


# Singleton instance
_extractor = StateExtractor()


def extract_from_env(env, obs_buf=None) -> torch.Tensor:
    """
    Convenience function to extract 134D state from live environment.

    Args:
        env: IsaacLab environment
        obs_buf: Optional observation buffer

    Returns:
        state_134d: (num_envs, 134) tensor
    """
    return _extractor.extract_from_env(env, obs_buf)


def extract_from_dataset(demo_group: h5py.Group) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to extract 134D states from HDF5 dataset.

    Args:
        demo_group: HDF5 demo group with states format

    Returns:
        states: (T, 134) state array
        actions: (T, action_dim) aligned actions
    """
    return _extractor.extract_from_dataset(demo_group)
