"""Custom reward and observation functions for Franka+Leap manipulation tasks."""

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


##
# Action penalty functions (matching Kuka Allegro's clamped versions)
##


def action_l2_clamped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the actions using L2 squared kernel with clamping.

    This matches Kuka Allegro's implementation to prevent exploding penalties
    during early random exploration.
    """
    return torch.sum(torch.square(env.action_manager.action), dim=1).clamp(-1000, 1000)


def action_rate_l2_clamped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel with clamping.

    This matches Kuka Allegro's implementation to prevent exploding penalties
    during early random exploration.
    """
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1).clamp(-1000, 1000)


##
# obs functions
##


def target_object_position_only(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Extract only the 3D position from the pose command, ignoring orientation.

    This is useful for lift tasks where orientation doesn't matter. The full pose command
    is 7D (x, y, z, qw, qx, qy, qz), but we only need the first 3 elements.

    Args:
        env: The RL environment.
        command_name: Name of the command to extract from.

    Returns:
        Position tensor of shape (num_envs, 3).
    """
    # Get the full 7D command (position + quaternion)
    full_command = env.command_manager.get_command(command_name)
    # Return only the position (first 3 elements)
    return full_command[:, :3]


##
# Reward functions
##


def fingertips_to_object_distance(
    env: ManagerBasedRLEnv,
    std: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward for bringing fingertips close to the object (power grasp).

    Computes the average distance from all Leap hand fingertips to the object center.
    Uses exponential reward to encourage close proximity for power grasping.

    Args:
        env: The RL environment.
        std: Standard deviation for the Gaussian kernel.
        robot_cfg: Configuration for the robot entity.
        object_cfg: Configuration for the object entity.

    Returns:
        Reward tensor of shape (num_envs,).
    """
    # Get robot and object
    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    # Use all 4 fingertips for power grasp
    fingertip_names = [
        "fingertip",        # Index finger
        "thumb_fingertip",  # Thumb
        "fingertip_2",      # Middle finger
        "fingertip_3",      # Ring finger
    ]

    # Get object position in world frame
    object_pos_w = object.data.root_pos_w  # (num_envs, 3)

    # Find fingertip body indices
    fingertip_indices = []
    for name in fingertip_names:
        try:
            # Find the body index for this fingertip
            body_idx = robot.find_bodies(name)[0][0]  # Returns (ids, names)
            fingertip_indices.append(body_idx)
        except:
            # If fingertip not found, skip it
            pass

    if len(fingertip_indices) == 0:
        # No fingertips found, return zero reward
        return torch.zeros(env.num_envs, device=env.device)

    # Get all fingertip positions
    fingertip_distances = []
    for idx in fingertip_indices:
        # Get fingertip position in world frame
        fingertip_pos_w = robot.data.body_pos_w[:, idx, :]  # (num_envs, 3)

        # Compute distance to object
        distance = torch.norm(object_pos_w - fingertip_pos_w, dim=-1)  # (num_envs,)
        fingertip_distances.append(distance)

    # Average distance across all fingertips
    avg_distance = torch.stack(fingertip_distances, dim=0).mean(dim=0)  # (num_envs,)

    # Exponential reward (closer = higher reward)
    return torch.exp(-avg_distance / std)


def fingertips_object_grasp_reward(
    env: ManagerBasedRLEnv,
    min_distance: float = 0.02,
    max_distance: float = 0.10,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward for having fingertips at appropriate grasping distance from object (power grasp).

    Rewards when all fingertips are at ideal distance for power grasping.
    Peak reward when all fingertips are at ideal grasping distance.

    Args:
        env: The RL environment.
        min_distance: Minimum ideal distance (too close = collision).
        max_distance: Maximum ideal distance (too far = no grasp).
        robot_cfg: Configuration for the robot entity.
        object_cfg: Configuration for the object entity.

    Returns:
        Reward tensor of shape (num_envs,).
    """
    # Get robot and object
    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    # Use all 4 fingertips for power grasp
    fingertip_names = [
        "fingertip",        # Index finger
        "thumb_fingertip",  # Thumb
        "fingertip_2",      # Middle finger
        "fingertip_3",      # Ring finger
    ]

    # Get object position in world frame
    object_pos_w = object.data.root_pos_w  # (num_envs, 3)

    # Find fingertip body indices
    fingertip_indices = []
    for name in fingertip_names:
        try:
            body_idx = robot.find_bodies(name)[0][0]
            fingertip_indices.append(body_idx)
        except:
            pass

    if len(fingertip_indices) == 0:
        return torch.zeros(env.num_envs, device=env.device)

    # Count how many fingertips are in the ideal range
    fingers_in_range = torch.zeros(env.num_envs, device=env.device)

    for idx in fingertip_indices:
        fingertip_pos_w = robot.data.body_pos_w[:, idx, :]
        distance = torch.norm(object_pos_w - fingertip_pos_w, dim=-1)

        # Check if in ideal grasping range
        in_range = (distance >= min_distance) & (distance <= max_distance)
        fingers_in_range += in_range.float()

    # Normalize by number of fingertips (0 to 1)
    return fingers_in_range / len(fingertip_indices)


def finger_closure_reward(
    env: ManagerBasedRLEnv,
    proximity_threshold: float = 0.15,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward for closing the hand fingers ONLY when near the object.

    Rewards when Leap hand joints are in a closed/grasping configuration,
    but ONLY if the hand is close to the object. This prevents the policy
    from learning to close fingers prematurely when far from the object.

    Args:
        env: The RL environment.
        proximity_threshold: Maximum distance (m) from object to reward closure.
        robot_cfg: Configuration for the robot entity.
        object_cfg: Configuration for the object entity.

    Returns:
        Reward tensor of shape (num_envs,).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    # Get palm position
    try:
        palm_idx = robot.find_bodies("palm_lower")[0][0]
    except:
        palm_idx = robot.find_bodies("panda_link7")[0][0]

    palm_pos_w = robot.data.body_pos_w[:, palm_idx, :]  # (num_envs, 3)
    object_pos_w = object.data.root_pos_w  # (num_envs, 3)

    # Compute distance from palm to object
    distance = torch.norm(palm_pos_w - object_pos_w, dim=-1)  # (num_envs,)

    # Only reward closure if hand is close to object
    is_near = distance < proximity_threshold  # (num_envs,)

    # Leap hand joints are indices 7-22 (after 7 Franka arm joints)
    # Joint positions typically range from 0 (open) to ~1.5 (closed)
    hand_joint_pos = robot.data.joint_pos[:, 7:]  # Shape: (num_envs, 16)

    # Average closure across all hand joints
    # Higher values = more closed = better grasp
    avg_closure = hand_joint_pos.mean(dim=1)  # Shape: (num_envs,)

    # Normalize to 0-1 range (assuming max closure ~1.5 rad)
    # Reward closure between 0.3-1.2 rad (partial to full grasp)
    normalized_closure = torch.clamp(avg_closure / 1.5, 0.0, 1.0)

    # Only give reward when near object
    return normalized_closure * is_near.float()


def object_hand_relative_velocity(
    env: ManagerBasedRLEnv,
    std: float = 0.1,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward for object moving with the hand (grasp quality proxy).

    When the object is properly grasped, its velocity should match the hand's velocity.
    This reward is high when object and hand move together, indicating a stable grasp.

    Args:
        env: The RL environment.
        std: Standard deviation for the Gaussian kernel.
        robot_cfg: Configuration for the robot entity.
        object_cfg: Configuration for the object entity.

    Returns:
        Reward tensor of shape (num_envs,).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    # Get palm body index (palm_lower is the base of the Leap hand)
    try:
        palm_idx = robot.find_bodies("palm_lower")[0][0]
    except:
        # Fallback to wrist if palm not found
        palm_idx = robot.find_bodies("panda_link7")[0][0]

    # Get velocities
    palm_vel_w = robot.data.body_vel_w[:, palm_idx, :3]  # Linear velocity (num_envs, 3)
    object_vel_w = object.data.root_lin_vel_w  # (num_envs, 3)

    # Compute relative velocity magnitude
    relative_vel = torch.norm(object_vel_w - palm_vel_w, dim=-1)  # (num_envs,)

    # Exponential reward (lower relative velocity = higher reward)
    # If object moves with hand, relative_vel ≈ 0, reward ≈ 1
    return torch.exp(-relative_vel / std)


def object_angular_velocity_penalty(
    env: ManagerBasedRLEnv,
    threshold: float = 2.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Penalty for object spinning/rotating (indicates poor grasp or flinging).

    A properly grasped object should have minimal angular velocity.
    High angular velocity indicates the object is spinning, tumbling, or being flung.

    Args:
        env: The RL environment.
        threshold: Angular velocity threshold (rad/s) for penalty.
        object_cfg: Configuration for the object entity.

    Returns:
        Penalty tensor of shape (num_envs,). Negative values.
    """
    object: RigidObject = env.scene[object_cfg.name]

    # Get object angular velocity in world frame
    object_ang_vel = object.data.root_ang_vel_w  # (num_envs, 3)

    # Compute magnitude of angular velocity
    ang_vel_mag = torch.norm(object_ang_vel, dim=-1)  # (num_envs,)

    # Penalty proportional to angular velocity above threshold
    # No penalty if below threshold, linear penalty above
    penalty = torch.clamp(ang_vel_mag - threshold, min=0.0)

    return -penalty


def fingertip_enclosure_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward for fingertips surrounding/enclosing the object from multiple directions.

    A good power grasp requires fingertips to be positioned around the object,
    not all on one side. This measures the spatial distribution of fingertips
    relative to the object center.

    Args:
        env: The RL environment.
        robot_cfg: Configuration for the robot entity.
        object_cfg: Configuration for the object entity.

    Returns:
        Reward tensor of shape (num_envs,).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    # Use all 4 fingertips
    fingertip_names = [
        "fingertip",        # Index finger
        "thumb_fingertip",  # Thumb
        "fingertip_2",      # Middle finger
        "fingertip_3",      # Ring finger
    ]

    # Get object position in world frame
    object_pos_w = object.data.root_pos_w  # (num_envs, 3)

    # Find fingertip body indices
    fingertip_indices = []
    for name in fingertip_names:
        try:
            body_idx = robot.find_bodies(name)[0][0]
            fingertip_indices.append(body_idx)
        except:
            pass

    if len(fingertip_indices) < 3:
        # Need at least 3 fingers for enclosure
        return torch.zeros(env.num_envs, device=env.device)

    # Collect relative positions (fingertip - object)
    relative_positions = []
    for idx in fingertip_indices:
        fingertip_pos_w = robot.data.body_pos_w[:, idx, :]  # (num_envs, 3)
        rel_pos = fingertip_pos_w - object_pos_w  # (num_envs, 3)
        # Normalize to unit vectors
        rel_pos_normalized = rel_pos / (torch.norm(rel_pos, dim=-1, keepdim=True) + 1e-6)
        relative_positions.append(rel_pos_normalized)

    # Stack all relative positions: (num_fingertips, num_envs, 3)
    relative_positions = torch.stack(relative_positions, dim=0)

    # High variance = fingers spread around object = good enclosure
    # Low variance = fingers clustered on one side = poor grasp
    position_variance = torch.var(relative_positions, dim=0).sum(dim=-1)  # (num_envs,)

    # Normalize to 0-1 range (empirically, good variance is ~0.5-1.5)
    # We want to reward variance, so higher is better
    normalized_variance = torch.clamp(position_variance / 1.0, 0.0, 1.0)

    return normalized_variance


def grasp_stability_reward(
    env: ManagerBasedRLEnv,
    vel_std: float = 0.1,
    ang_vel_threshold: float = 2.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Combined reward for stable grasping (object moves with hand, minimal spinning).

    This combines relative velocity matching and angular velocity penalty into
    a single metric for grasp stability.

    Args:
        env: The RL environment.
        vel_std: Standard deviation for relative velocity kernel.
        ang_vel_threshold: Angular velocity threshold for penalty.
        robot_cfg: Configuration for the robot entity.
        object_cfg: Configuration for the object entity.

    Returns:
        Reward tensor of shape (num_envs,).
    """
    # Get relative velocity reward (0 to 1)
    rel_vel_reward = object_hand_relative_velocity(env, std=vel_std, robot_cfg=robot_cfg, object_cfg=object_cfg)

    # Get angular velocity penalty (negative)
    ang_vel_penalty = object_angular_velocity_penalty(env, threshold=ang_vel_threshold, object_cfg=object_cfg)

    # Combine: high reward when moving together AND not spinning
    # Scale angular penalty to 0-1 range for combination
    ang_vel_bonus = torch.clamp(1.0 + ang_vel_penalty / 5.0, 0.0, 1.0)

    return rel_vel_reward * ang_vel_bonus


def object_orientation_penalty(
    env: ManagerBasedRLEnv,
    threshold: float = 0.5,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Penalty for object tilting/rotating away from upright orientation.

    Penalizes when the object deviates from its initial upright orientation.
    This prevents excessive rotation during grasping and lifting.

    Args:
        env: The RL environment.
        threshold: Angular deviation threshold (radians) before penalty applies.
        object_cfg: Configuration for the object entity.

    Returns:
        Penalty tensor of shape (num_envs,). Negative values.
    """
    object: RigidObject = env.scene[object_cfg.name]

    # Get object orientation quaternion (w, x, y, z)
    object_quat = object.data.root_quat_w  # (num_envs, 4)

    # Using the quaternion's w component: angle = 2 * acos(|w|)
    w = object_quat[:, 0]  # (num_envs,)

    # Angle from upright (0 to pi radians)
    tilt_angle = 2.0 * torch.acos(torch.clamp(torch.abs(w), -1.0, 1.0))  # (num_envs,)

    # Penalty proportional to tilt beyond threshold
    # No penalty if upright (< threshold), linear penalty beyond
    penalty = torch.clamp(tilt_angle - threshold, min=0.0)

    return -penalty


def object_goal_distance_gated(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    gate_threshold: float = 0.1,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward for tracking goal position, GATED by proximity to object.

    Similar to standard object_goal_distance, but only gives reward when
    fingertips are close to the object. This prevents "air tracking" behavior
    where the agent moves toward the goal without actually grasping.

    This mimics Kuka Allegro's contact-gated rewards without requiring contact sensors.

    Args:
        env: The RL environment.
        std: Standard deviation for the tanh kernel.
        minimal_height: Minimum object height to receive reward.
        command_name: Name of the pose command.
        gate_threshold: Maximum distance (m) from object to enable reward.
        robot_cfg: Configuration for the robot entity.
        object_cfg: Configuration for the object entity.

    Returns:
        Reward tensor of shape (num_envs,).
    """
    from isaaclab.utils.math import combine_frame_transforms

    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)

    # Compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)

    # Distance from object to goal
    distance_to_goal = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)

    # Height gate: object must be lifted
    height_gate = object.data.root_pos_w[:, 2] > minimal_height

    # Proximity gate: Check if fingertips are close to object
    fingertip_names = ["fingertip", "thumb_fingertip", "fingertip_2", "fingertip_3"]
    object_pos_w = object.data.root_pos_w  # (num_envs, 3)

    # Find minimum distance from any fingertip to object
    min_fingertip_distance = torch.ones(env.num_envs, device=env.device) * 1e6
    for name in fingertip_names:
        try:
            body_idx = robot.find_bodies(name)[0][0]
            fingertip_pos_w = robot.data.body_pos_w[:, body_idx, :]
            distance = torch.norm(object_pos_w - fingertip_pos_w, dim=-1)
            min_fingertip_distance = torch.min(min_fingertip_distance, distance)
        except:
            pass

    # Gate: Only reward when close to object (mimics contact gating)
    proximity_gate = min_fingertip_distance < gate_threshold

    # Combined gate: must be lifted AND near object
    gate = height_gate & proximity_gate

    # Compute reward
    reward = 1 - torch.tanh(distance_to_goal / std)

    # Apply gate
    return reward * gate.float()


def success_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    pos_std: float,
    rot_std: float | None = None,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward success by comparing commanded pose to the object pose using tanh kernels on error.

    This is a large bonus reward for getting the object close to the goal position.
    Matches the Kuka Allegro success reward implementation.

    Args:
        env: The RL environment.
        command_name: Name of the pose command.
        pos_std: Standard deviation for position error tanh kernel.
        rot_std: Standard deviation for rotation error tanh kernel (None for position-only tasks).
        robot_cfg: Configuration for the robot entity.
        object_cfg: Configuration for the object entity.

    Returns:
        Reward tensor of shape (num_envs,).
    """
    from isaaclab.utils.math import combine_frame_transforms, compute_pose_error

    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)

    # Compute desired pose in world frame
    des_pos_w, des_quat_w = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3], command[:, 3:7]
    )

    # Compute pose error
    pos_err, rot_err = compute_pose_error(des_pos_w, des_quat_w, object.data.root_pos_w, object.data.root_quat_w)
    pos_dist = torch.norm(pos_err, dim=1)

    if not rot_std:
        # Position-only task (like lift) - square helps normalize reward magnitude
        return (1 - torch.tanh(pos_dist / pos_std)) ** 2

    # Position + orientation task
    rot_dist = torch.norm(rot_err, dim=1)
    return (1 - torch.tanh(pos_dist / pos_std)) * (1 - torch.tanh(rot_dist / rot_std))
