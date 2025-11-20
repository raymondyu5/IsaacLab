import torch
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math
import isaaclab.envs.mdp as mdp

from pathlib import Path
from dataclasses import dataclass, field
from isaaclab.utils import configclass
from isaaclab.assets import Articulation, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.managers import ObservationTermCfg as ObsTerm
# from isaaclab.utils .math import subtract_frame_transforms

from scripts.workflows.utils.robot_cfg import WIDOWX_CFG
from isaaclab.sim.spawners import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


@configclass
class BridgeKitchenUWSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75),
                                     intensity=1000.0),
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.0]),
        spawn=sim_utils.GroundPlaneCfg(),
        collision_group=-1,
    )
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, -0.0],
                                                rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(
            usd_path=
            f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
        ),
    )

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # # end-effector sensor: will be populated by agent env cfg
        # ee_frame: FrameTransformerCfg = MISSING
        # Listens to the required transforms
        # robot = WIDOWX_CFG
        # marker_cfg = FRAME_MARKER_CFG.copy()
        # marker_cfg.markers["frame"].scale = (0.03, 0.03, 0.03)
        # marker_cfg.prim_path = "/Visuals/FrameTransformer"
        # self.ee_frame = FrameTransformerCfg(
        #     prim_path="{ENV_REGEX_NS}/robot/wx250s/wx250s_base_link",
        #     debug_vis=True,
        #     visualizer_cfg=marker_cfg,
        #     target_frames=[
        #         FrameTransformerCfg.FrameCfg(
        #             prim_path=
        #             "{ENV_REGEX_NS}/robot/wx250s/wx250s_ee_gripper_link",
        #             name="end_effector",
        #             offset=OffsetCfg(pos=[0.0, 0.0, 0.0],
        #                              rot=[0.0, -0.7, 0.0, 0.7]),
        #         ),
        #     ],
        # )


@configclass
class ActionCfg:
    arm_action: mdp.JointPositionActionCfg = None
    gripper_action: mdp.BinaryJointPositionActionCfg = None


def get_jp(
        env,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    gripper_state = robot.data.joint_pos
    return gripper_state


def get_camera_data(env: ManagerBasedEnv,
                    camera_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
                    type: str = "rgb") -> torch.Tensor:

    camera = env.scene[camera_cfg.name]
    return camera.data.output[type][0][..., :3]


def robottip_pose(env):
    # obtain quantities from simulation

    ee_pose_w = env.scene[
        "wx250s_ee_gripper_link"].data.root_link_state_w[:, :7]
    ee_pose_w[:, :3] -= env.scene.env_origins

    gripper = env.scene["robot"]._data.joint_pos[:, -1][:, None]

    return torch.cat([ee_pose_w, gripper], dim=1)


@configclass
class ObservationCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy."""

        # control_joint_action = ObsTerm(func=mdp.control_joint_action)
        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos)
        ee_pose = ObsTerm(func=robottip_pose)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # reset_robot = EventTerm(func=reset_ee_pose, mode="reset")


@configclass
class CommandsCfg:
    """Command terms for the MDP."""


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    """Curriculum configuration."""


@configclass
class BridgeKitchenUWCfg(ManagerBasedEnvCfg):
    env_config: str = field(default=None)

    scene = BridgeKitchenUWSceneCfg(num_envs=1, env_spacing=2.0)

    actions = ActionCfg()
    rewards = RewardsCfg()
    observations = ObservationCfg()

    terminations = TerminationsCfg()
    commands = CommandsCfg()
    events = EventCfg()
    curriculum = CurriculumCfg()

    def __post_init__(self):
        self.episode_length_s = 15

        self.viewer.eye = (-4.5, 0.0, 3.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)

        self.decimation = 4
        self.sim.dt = 1 / 60
        self.sim.render_interval = 12

        self.sim.physx.gpu_temp_buffer_capacity = 2**30
        self.sim.physx.gpu_heap_capacity = 2**30
        self.sim.physx.gpu_collision_stack_size = 2**30

        self.instruction = "put the eggplant in the sink"
