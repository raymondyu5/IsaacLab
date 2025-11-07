"""Franka + Leap Hand pinch grasping environment configuration.

This configuration is designed for a reach + pinch task where the robot learns to:
1. Reach toward the object
2. Pinch the object using only the thumb and index finger

Robot Configuration:
    - Franka arm joints: panda_joint1 through panda_joint7 (7 DOF)
    - Leap hand joints: j0 through j15 (16 DOF)
    - Total DOF: 23

Contact Sensor Link Names (Pinch grasp uses only):
    - fingertip       (index finger)
    - thumb_fingertip (thumb)

Rewards focused on:
    - Reaching with thumb and index fingertips toward object
    - Making contact with both thumb and index
    - Maintaining good pinch contact during manipulation
"""

from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg, RewardTermCfg as RewTerm

from .joint_pos_env_franka_leap_cfg import FrankaLeapCubeLiftEnvCfg
from .mdp import rewards as mdp_rewards


@configclass
class FrankaLeapPinchEnvCfg(FrankaLeapCubeLiftEnvCfg):
    """Configuration for Franka Panda with Leap Hand pinch grasping a cube.

    This inherits from the full lifting configuration but modifies the rewards
    to focus only on thumb and index finger contact/approach.
    """

    def __post_init__(self):
        super().__post_init__()

        # Override the finger contact reward to use pinch-specific version
        self.rewards.good_finger_contact = RewTerm(
            func=mdp_rewards.pinch_grasp_contacts,
            weight=0.5,
            params={"threshold": 1.0},
        )

        # Override the fingertips approach reward to use only thumb and index
        self.rewards.fingertips_approach_when_near = RewTerm(
            func=mdp_rewards.pinch_fingertips_approach_object_when_near,
            weight=1.5,
            params={
                "palm_distance_threshold": 0.15,
                "std": 0.05,
                "robot_cfg": SceneEntityCfg("robot", body_names=["palm_lower", "fingertip", "thumb_fingertip"]),
            },
        )

        # Update fingers_to_object to only consider thumb and index
        self.rewards.fingers_to_object = RewTerm(
            func=mdp_rewards.object_ee_distance,
            weight=1.0,
            params={
                "std": 0.4,
                "asset_cfg": SceneEntityCfg("robot", body_names=["palm_lower", "fingertip", "thumb_fingertip"]),
                "object_cfg": SceneEntityCfg("object"),
            },
        )

        # Update position tracking to use pinch contact check
        self.rewards.position_tracking = RewTerm(
            func=mdp_rewards.pinch_position_command_error_tanh,
            weight=2.0,
            params={
                "std": 0.2,
                "command_name": "object_pose",
            },
        )


@configclass
class FrankaLeapPinchEnvCfg_PLAY(FrankaLeapPinchEnvCfg):
    """Play configuration for Franka+Leap pinch task (fewer envs, no randomization)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        # Set difficulty to max to enable full gravity and all domain randomization
        self.curriculum.adr.params["init_difficulty"] = self.curriculum.adr.params["max_difficulty"]
