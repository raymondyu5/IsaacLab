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


def spanwn_robot(
    scene,
    main_path,
    spawn_list,
):
    for index, link_name in enumerate(spawn_list):

        mesh_name = link_name
        rigid_cfg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/" + f"{main_path}/{mesh_name}",
            spawn=None,
        )

        setattr(scene, f"{link_name}", rigid_cfg)


def config_robot(scene, env_cfg):

    spanwn_robot(scene, "Robot",
                 env_cfg["params"]["spawn_robot"]["spawn_arm_list"])
