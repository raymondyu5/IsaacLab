# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.simulation_cfg import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise

import isaaclab_tasks.manager_based.manipulation.inhand.mdp as mdp
from scripts.workflows.utils.config_setting import set_from_config
from isaaclab.sim.spawners import GroundPlaneCfg, UsdFileCfg
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
##
# Scene definition
##


@configclass
class InHandObjectSceneCfg(InteractiveSceneCfg):
    """Configuration for a scene with an object and a dexterous hand."""

    # # robots
    # robot: ArticulationCfg = MISSING

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.3, -0.0, -0.0],
                                                rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(
            usd_path="/home/wph52/IsaacLab/source/assets/table/table_instanceable.usd",
            scale=[1.0, 1.0, 1.0]),
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.0]),
        spawn=sim_utils.GroundPlaneCfg(size=(1000.0, 1000.0)),
        collision_group=-1,
    )
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(
            intensity=3000.0,
            color=(0.75, 0.75, 0.75),
        ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class KinematicObsGroupCfg(ObsGroup):
        """Observations with full-kinematic state information.

        This does not include acceleration or force information.
        """

        #observation terms (order preserved)
        #-- robot terms
        # joint_pos = ObsTerm(func=mdp.joint_pos, noise=Gnoise(std=0.005))
        # joint_vel = ObsTerm(func=mdp.joint_vel_rel,
        #                     scale=0.2,
        #                     noise=Gnoise(std=0.01))

        # -- action terms
        # last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    # observation groups
    policy: KinematicObsGroupCfg = KinematicObsGroupCfg()


@configclass
class EventCfg:
    """Configuration for randomization."""
    # reset_scene_to_default = EventTerm(func=mdp.reset_scene_to_default,
    #                                    mode="reset")


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- penalties
    # joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-2.5e-5)
    # action_l2 = RewTerm(func=mdp.action_l2, weight=-0.0001)
    # action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""


@configclass
class InHandObjectEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the in hand reorientation environment."""

    # Scene settings
    scene: InHandObjectSceneCfg = InHandObjectSceneCfg(num_envs=2,
                                                       env_spacing=2)
    # Simulation settings
    sim: SimulationCfg = SimulationCfg(
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=2**20,
            gpu_max_rigid_patch_count=2**23,
        ),
    )
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 3
        self.episode_length_s = 10
        # simulation settings
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation
        # change viewer settings
        self.viewer.eye = (30, 0.0, 1.5)
        self.viewer.lookat = [0.0, 0.0, 0.8]
        self.scene.replicate_physics = False
