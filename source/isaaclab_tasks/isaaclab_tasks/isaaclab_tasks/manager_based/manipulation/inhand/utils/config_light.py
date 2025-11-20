from isaaclab_assets import *

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.assets import RigidObjectCfg, DeformableObjectCfg, AssetBaseCfg, ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
import torch
from isaaclab.sensors.contact_sensor import ContactSensor, ContactSensorCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from source.isaaclab_tasks.isaaclab_tasks.manager_based.manipulation.inhand.mdp.obs import get_contact_obs_func
from isaaclab.managers import SceneEntityCfg
import random
from pxr import Gf

from isaaclab.sim.utils import clone, safe_set_attribute_on_usd_prim


def configure_light(object, scene_cfg):
    """
    Configure the light in the scene.
    """
    # Set the light color and intensity
    reset_func = EventTerm(func=randomize_scene_lighting_domelight,
                           mode="reset",
                           params={
                               "intensity_range": (1000.0, 5000.0),
                               "color_range":
                               ((0.7, 0.7, 0.7), (1.0, 1.0, 1.0))
                           })
    setattr(object.events, f"reset_light", reset_func)


def randomize_scene_lighting_domelight(
        env,
        env_ids: torch.Tensor,
        intensity_range: tuple[float, float],
        color_range: tuple[tuple[float, float, float], tuple[float, float,
                                                             float]],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("light"),
):
    asset = env.scene[asset_cfg.name]
    # num_lights = len(asset.prims)

    # env_orgin = env.scene.env_origins.clone()
    # env_orgin[..., 2] = 3.0
    # asset.set_world_poses(env_orgin)
    # for i in range(num_lights):
    light_prim = asset.prims[0]

    # Sample new light intensity
    new_intensity = random.uniform(*intensity_range)

    # Sample new light color
    new_color = [
        random.uniform(color_range[0][i], color_range[1][i]) for i in range(3)
    ]

    # Set attributes
    light_prim.GetAttribute("inputs:intensity").Set(new_intensity)
    light_prim.GetAttribute("inputs:color").Set(
        Gf.Vec3f(*new_color))  # Note: Gf.Vec3f from pxr.Gf
    # safe_set_attribute_on_usd_prim(light_prim,
    #                                "inputs:color",
    #                                new_color,
    #                                camel_case=True)
