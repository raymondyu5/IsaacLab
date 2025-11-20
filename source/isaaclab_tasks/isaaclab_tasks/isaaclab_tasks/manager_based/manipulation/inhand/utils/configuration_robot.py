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

init_state = ArticulationCfg.InitialStateCfg(pos=(0, 0, -0.0),
                                             rot=(1, 0, 0, 0),
                                             joint_pos=None)

robot_spawn_template = sim_utils.UsdFileCfg(
    usd_path=None,
    activate_contact_sensors=True,
    rigid_props=sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=True,
        max_depenetration_velocity=5.0,
    ),
    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=False,
        solver_position_iteration_count=1,
        solver_velocity_iteration_count=0),
)


def config_rel_jonit_arm(arm_type,
                         hand_type,
                         actions,
                         add_right_hand=False,
                         add_left_hand=False,
                         use_relative_mode=False,
                         ee_name=None):
    arm_actuator, arm_joint_pos, arm_action, arm_joint_limits = config_arm(
        arm_type)

    joint_names = list(arm_joint_pos.keys())
    arm_action = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=joint_names,
    )
    hand_actuator, hand_joint_pos, hand_action = config_hand(
        hand_type, arm_type)
    if add_left_hand:
        left_arm_action = arm_action.replace(asset_name="left_hand", )
        robot_hand_action = hand_action.replace(asset_name=f"left_hand")
        setattr(actions, f"left_hand_action", robot_hand_action)

        setattr(actions, "left_arm_action", left_arm_action)
    if add_right_hand:
        right_arm_action = arm_action.replace(asset_name="right_hand", )
        setattr(actions, "right_arm_action", right_arm_action)
        robot_hand_action = hand_action.replace(asset_name=f"right_hand")
        setattr(actions, f"right_hand_action", robot_hand_action)


def config_ik_arm(config, actions, events, use_relative_mode=False):

    arm_type = config["params"]["arm_type"]

    add_right_hand = config["params"]["add_right_hand"]
    add_left_hand = config["params"]["add_left_hand"]
    ee_name = config["params"].get("ee_name", "ee_link")
    arm_actuator, arm_joint_pos, arm_action, arm_joint_limits = config_arm(
        arm_type)

    joint_names = list(arm_joint_pos.keys())
    arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=joint_names,
        body_name=ee_name,
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=use_relative_mode,
            ik_method="dls"),
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
            pos=[0.0, 0.0, 0.0]),
    )
    hand_names = []
    if add_left_hand:
        left_arm_action = arm_action.replace(asset_name="left_hand", )
        hand_names.append("left_hand")

        setattr(actions, "left_arm_action", left_arm_action)
    if add_right_hand:
        right_arm_action = arm_action.replace(asset_name="right_hand", )
        setattr(actions, "right_arm_action", right_arm_action)
        hand_names.append("right_hand")
    if config["params"].get("real_eval_mode", False):
        return
    if not use_relative_mode:

        for name in hand_names:

            reset_pole_position = EventTerm(
                func=mdp.randomize_robot_ee_pose,
                mode="reset",
                params={
                    "asset_name":
                    name,
                    "pose_range":
                    config["params"].get("random_ee_pose_range",
                                         (-0.06, 0.06)),
                    "init_ee_pose":
                    torch.as_tensor(config["params"]["init_ee_pose"]),
                },
            )
            setattr(events, f"reset_{name}_ee_pose", reset_pole_position)


def config_env(
    object,
    arm_actuator,
    hand_actuator,
    arm_action,
    hand_action,
    init_robot_state,
    side_name="robot",
    enabled_self_collisions=False,
    robot_usd_file=None,
):

    robot_spawn = robot_spawn_template.replace(
        usd_path=robot_usd_file,
        semantic_tags=[("class", f"{side_name}_hand")],
    )
    robot_spawn.articulation_props.enabled_self_collisions = enabled_self_collisions

    robot = ArticulationCfg(
        spawn=robot_spawn,
        init_state=init_robot_state,
        soft_joint_pos_limit_factor=1.0,
    )
    robot.actuators = arm_actuator | hand_actuator

    if arm_action is not None:
        robot_arm_action = arm_action.replace(asset_name=f"{side_name}_hand")
        setattr(object.actions, f"{side_name}_arm_action", robot_arm_action)

    robot = robot.replace(prim_path="{ENV_REGEX_NS}" + f"/{side_name}_hand")
    setattr(object.scene, f"{side_name}_hand", robot)

    robot_hand_action = hand_action.replace(asset_name=f"{side_name}_hand")
    setattr(object.actions, f"{side_name}_hand_action", robot_hand_action)

    joint_pos = ObsTerm(func=mdp.joint_pos,
                        params={"asset_name": f"{side_name}_hand"})
    setattr(object.observations.policy, f"{side_name}_hand_joint_pos",
            joint_pos)


def spanwn_robot_hand(
    scene,
    main_path,
    spawn_list,
    hand_side=None,
):
    for index, link_name in enumerate(spawn_list):

        link_name = link_name.replace(".*", hand_side)

        mesh_name = link_name
        rigid_cfg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/" + f"{main_path}/{mesh_name}",
            spawn=None,
        )

        setattr(scene, f"{hand_side}_{link_name}", rigid_cfg)


def config_reset_robot_pose(
    events,
    env_cfg,
    hand_side=None,
):
    # hand_pose = torch.as_tensor(env_cfg["params"][f"{hand_side}_robot_pose"])

    # reset = EventTerm(func=mdp.reset_rigid_articulation,
    #                   mode="reset",
    #                   params={
    #                       "target_name": f"{hand_side}_hand",
    #                       "pose_range": {
    #                           "z": (-0.20, 0.0),
    #                       }
    #                   })
    # setattr(events, f"reset_{hand_side}_robot_root_pose", reset)
    hand_pose = torch.as_tensor(env_cfg["params"][f"{hand_side}_robot_pose"])
    hand_pose = EventTerm(func=mdp.reset_robot_root_state,
                          mode="reset",
                          params={
                              "root_pose": hand_pose,
                              "asset_name": f"{hand_side}_hand",
                          })
    setattr(events, f"reset_{hand_side}_robot_root_pose", hand_pose)


def config_reset_robot_setting(
    events,
    env_cfg,
    joints_info,
    arm_info,
    hand_side=None,
):

    if env_cfg["params"].get("real_eval_mode", False):
        return

    if env_cfg["params"].get("teleop_mode", False):
        return

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        # min_step_count_between_reset=800,
        params={
            "asset_cfg": SceneEntityCfg(f"{hand_side}_hand"),
            "static_friction_range": (0.5, 1.5),
            "dynamic_friction_range": (0.5, 1.5),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )

    setattr(events, f"reset_{hand_side}_robot_physics_material",
            robot_physics_material)

    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        # min_step_count_between_reset=800,
        mode="reset",
        params={
            "asset_cfg":
            SceneEntityCfg(f"{hand_side}_hand",
                           joint_names=list(joints_info.keys())),
            "stiffness_distribution_params": (0.8, 1.2),  # default: 3.0
            "damping_distribution_params": (0.8, 1.2),  # default: 0.1
            "operation":
            "scale",
            "distribution":
            "uniform",
        },
    )
    setattr(events, f"reset_{hand_side}_robot_joint_stiffness_and_damping",
            robot_joint_stiffness_and_damping)

    external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(0.0, 10.0),
        params={
            "asset_cfg": SceneEntityCfg(f"{hand_side}_hand"),
            "force_range": (-0.25, 0.25),
            "torque_range": (-0.25, 0.25),
        },
    )
    setattr(events, f"reset_{hand_side}_robot_external_force_torque",
            external_force_torque)

    robot_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        # min_step_count_between_reset=800,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(f"{hand_side}_hand", ),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    setattr(events, f"reset_{hand_side}_robot_scale_mass", robot_scale_mass)

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        # min_step_count_between_reset=800,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(f"{hand_side}_hand"),
            "static_friction_range": (0.5, 1.5),
            "dynamic_friction_range": (0.5, 1.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
        },
    )
    setattr(events, f"reset_{hand_side}_robot_physics_material",
            physics_material)

    # reset_pole_position = EventTerm(
    #     func=mdp.reset_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg(f"{hand_side}_hand", ),
    #         "position_range": (-0.05 * math.pi, 0.05 * math.pi),
    #         "velocity_range": (-0.01 * math.pi, 0.01 * math.pi),
    #     },
    # )
    # setattr(events, f"reset_{hand_side}_robot_pole_position",
    #         reset_pole_position)

    # reset_pole_position = EventTerm(
    #     func=mdp.reset_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg(f"{hand_side}_hand", ),
    #         "position_range": (-0.1 * math.pi, 0.1 * math.pi),
    #         "velocity_range": (-0.02 * math.pi, 0.02 * math.pi),
    #     },
    # )

    reset_pole_position = EventTerm(
        func=mdp.reset_joints_by_interpolate,
        mode="reset",
        params={
            "asset_cfg":
            SceneEntityCfg(f"{hand_side}_hand", ),
            "position_range": (-0.0 * math.pi, 0.0 * math.pi),
            "interpolation_range":
            torch.as_tensor(env_cfg["params"]["joint_interpolate_region"]),
        },
    )
    setattr(events, f"reset_{hand_side}_robot_pole_position",
            reset_pole_position)


def reset_joints_limits(
    env,
    env_ids: torch.Tensor,
    joint_limits: tuple[float, float],
    asset_name: str = "robot",
):
    """Reset the robot joints with offsets around the default position and velocity by the given ranges.

    This function samples random values from the given ranges and biases the default joint positions and velocities
    by these values. The biased values are then set into the physics simulation.
    """
    # extract the used quantities (to enable type-hinting)
    robot = env.scene[asset_name]

    robot_limit = robot._data.joint_limits.clone()

    robot_limit[:, :7, :] = joint_limits.to(env.device)

    robot.write_joint_position_limit_to_sim(robot_limit, env_ids=env_ids)


def config_reset_joint_pose(events,
                            arm_joint_pos,
                            num_hand_joints,
                            arm_joint_limits,
                            hand_side=None):

    target_joint_value = torch.cat(
        [torch.as_tensor(arm_joint_pos),
         torch.zeros((num_hand_joints))], )

    reset_joint = EventTerm(func=mdp.reset_joints_by_values,
                            mode="reset",
                            params={
                                "joint_pose":
                                torch.as_tensor(target_joint_value),
                                "asset_name": f"{hand_side}_hand",
                            })
    setattr(events, f"reset_{hand_side}_hand_joint_pos", reset_joint)

    reset_joint_limits = EventTerm(func=reset_joints_limits,
                                   mode="reset",
                                   params={
                                       "asset_name":
                                       f"{hand_side}_hand",
                                       "joint_limits":
                                       torch.as_tensor(
                                           list(arm_joint_limits.values())),
                                   })
    setattr(events, f"reset_{hand_side}_hand_joint_limits", reset_joint_limits)


def config_bimanual_robot_contact_sensor(scene, env_cfg, object_name,
                                         hand_side):

    main_path = env_cfg["params"]["contact_sensor"][f"{hand_side}_hand"][
        "main_path"]
    spawn_list = env_cfg["params"]["contact_sensor"]["spawn_contact_list"]
    filter_prim_paths_expr = []
    for index, name in enumerate(object_name):
        filter_prim_paths_expr += ["{ENV_REGEX_NS}" + f"/{name}"]

    for link_name in spawn_list:

        mesh_name = f"{link_name}"
        contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}" + f"/{main_path}/{mesh_name}",
            update_period=0.0,
            history_length=3,
            filter_prim_paths_expr=filter_prim_paths_expr,
            debug_vis=False)
        setattr(scene, f"{hand_side}_{link_name}_contact", contact_sensor)


def config_robot_contact_sensor(scene, env_cfg, hand_side):

    main_path = env_cfg["params"]["contact_sensor"][f"{hand_side}_hand"][
        "main_path"]
    spawn_list = env_cfg["params"]["contact_sensor"]["spawn_contact_list"]

    object_name = env_cfg["params"]["spawn_multi_assets_name"]
    filter_prim_paths_expr = ["{ENV_REGEX_NS}" + f"/{object_name}"]
    if env_cfg["params"].get("articulation_links", None) is not None:
        articulation_links = env_cfg["params"]["articulation_links"]
        filter_prim_paths_expr += [
            "{ENV_REGEX_NS}" + f"/{object_name}/{articulation_links}"
        ]

    for link_name in spawn_list:

        mesh_name = f"{link_name}"
        contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}" + f"/{main_path}/{mesh_name}",
            update_period=0.0,
            history_length=3,
            filter_prim_paths_expr=filter_prim_paths_expr,
            debug_vis=False)
        setattr(scene, f"{hand_side}_{link_name}_contact", contact_sensor)


def config_robot_contact_obs(object, env_cfg, hand_side):

    contact_sensor = env_cfg["params"]["contact_sensor"]["spawn_contact_list"]

    contact_obs = ObsTerm(func=get_contact_obs_func,
                          params={
                              "asset_name": f"{hand_side}_hand",
                              "sensor_name": contact_sensor,
                              "hand_side": hand_side,
                          })
    setattr(object.observations.policy, f"{hand_side}_contact_obs",
            contact_obs)


def config_usd(arm_type, hand_type, enable_sensor=False):
    if arm_type == "xarm5":
        if hand_type == "leap":
            right_robot_usd_file = f"source/assets/robot/leaf_xarm_v2/raw_right_leap_xarm.usd"
            left_robot_usd_file = f"source/assets/robot/leaf_xarm_v2/raw_left_leap_xarm.usd"
        if hand_type == "ability":
            right_robot_usd_file = f"source/assets/robot/ability_hand/raw_right_ability.usd"
            left_robot_usd_file = f"source/assets/robot/ability_hand/raw_left_ability.usd"

    if arm_type == "xarm7":
        if hand_type == "leap":
            right_robot_usd_file = f"source/assets/robot/xarm7_leap/raw_xarm7_right_leap.usd"
            left_robot_usd_file = f"source/assets/robot/xarm7_leap/raw_xarm7_left_leap.usd"
        if hand_type == "ability":
            right_robot_usd_file = f"source/assets/robot/xarm7_ability/raw_xarm7_right_ability.usd"
            left_robot_usd_file = f"source/assets/robot/xarm7_ability/raw_xarm7_left_ability.usd"

    if arm_type == "ur5e":
        if hand_type == "leap":
            if enable_sensor:
                right_robot_usd_file = f"source/assets/robot/ur5e_leap/raw_ur5e_right_leap_sensor.usd"
                left_robot_usd_file = f"source/assets/robot/ur5e_leap/raw_ur5e_left_leap_sensor.usd"
            else:
                right_robot_usd_file = f"source/assets/robot/ur5e_leap/raw_ur5e_right_leap.usd"
                left_robot_usd_file = f"source/assets/robot/ur5e_leap/raw_ur5e_left_leap.usd"
        if hand_type == "ability":
            right_robot_usd_file = f"source/assets/robot/ur5e_ability/raw_ur5e_right_ability.usd"
            left_robot_usd_file = f"source/assets/robot/ur5e_ability/raw_ur5e_left_ability.usd"
    if arm_type == "franka":
        if hand_type == "leap":
            right_robot_usd_file = f"source/assets/robot/franka_leap/franka_right_leap.usd"
            left_robot_usd_file = f"source/assets/robot/franka_leap/franka_left_leap_long_finger.usd"

    if arm_type is None:
        if hand_type == "leap":
            if enable_sensor:
                right_robot_usd_file = f"source/assets/robot/leap_hand_v2/raw_right_hand_sensor.usd"
                left_robot_usd_file = f"source/assets/robot/leap_hand_v2/raw_left_hand_sensor.usd"
            else:
                right_robot_usd_file = f"source/assets/robot/leap_hand_v2/free_right_hand.usd"
                left_robot_usd_file = f"source/assets/robot/leap_hand_v2/free_left_hand.usd"
        if hand_type == "ability":
            right_robot_usd_file = f"source/assets/robot/ability_hand/raw_right_ability.usd"
            left_robot_usd_file = f"source/assets/robot/ability_hand/raw_left_ability.usd"

    return {
        "right": right_robot_usd_file,
        "left": left_robot_usd_file,
    }


def config_arm(arm_type):
    if arm_type == "xarm7":
        arm_actuator = XARM7_ACTUATOR_CFG
        arm_joint_pos = XARM7_DEFAULT_JOINT_POS
        arm_action = XARM7_ARM_ACTION

    elif arm_type == "xarm5":
        arm_actuator = XARM5_ACTUATOR_CFG
        arm_joint_pos = XARM5_DEFAULT_JOINT_POS
        arm_action = XARM5_ARM_ACTION

    elif arm_type == "ur5e":
        arm_actuator = UR5_ARM_ACTUATOR_CFG
        arm_joint_pos = UR5_DEFAULT_JOINT_POS
        arm_action = UR5_ARM_ACTION

    elif arm_type == "franka":
        arm_actuator = FRANKA_ARM_ACTUATOR_HIGH_PD_CFG
        arm_joint_pos = FRANKA_DEFAULT_JOINT_POS
        arm_action = FRANKA_ARM_ACTION
        arm_joint_limits = FRANKA_JOINT_LIMITS

    elif arm_type is None:
        arm_actuator = FREE_LEAP_ARM_ACTUATOR_CFG
        arm_joint_pos = FREE_LEAP_ARM_DEFAULT_JOINT_POS
        arm_action = FREE_LEAP_ARM_ACTION
    else:
        raise ValueError(f"Unknown arm type: {arm_type}")
    return arm_actuator, arm_joint_pos, arm_action, arm_joint_limits


def config_hand(hand_type, arm_type):
    if hand_type == "ability":
        hand_actuator = ABILITY_HAND_ACTUATOR_CFG
        hand_joint_pos = ABILITY_DEFAULT_JOINT_POS
        hand_action = ABILITY_HAND_ACTION
    elif hand_type == "leap":

        hand_actuator = LEAP_HAND_ACTUATOR_CFG
        hand_joint_pos = LEAP_DEFAULT_JOINT_POS
        hand_action = LEAP_HAND_ACTION

    return hand_actuator, hand_joint_pos, hand_action


def config_robot(object, actions, env_cfg):

    arm_type = env_cfg["params"]["arm_type"]
    hand_type = env_cfg["params"]["hand_type"]
    add_right_hand = env_cfg["params"]["add_right_hand"]
    add_left_hand = env_cfg["params"]["add_left_hand"]
    enable_sensor = env_cfg["params"].get("enable_sensor", False)
    enabled_self_collisions = env_cfg["params"].get("enabled_self_collisions",
                                                    False)

    arm_actuator, arm_joint_pos, arm_action, arm_joint_limits = config_arm(
        arm_type)
    hand_actuator, hand_joint_pos, hand_action = config_hand(
        hand_type, arm_type)
    robot_usd_files = config_usd(arm_type, hand_type, enable_sensor)

    robot_usd_files["right"] = env_cfg["params"].get("right_robot_usd",
                                                     robot_usd_files["right"])
    robot_usd_files["left"] = env_cfg["params"].get("right_robot_usd",
                                                    robot_usd_files["left"])
    spawn_robot = env_cfg["params"]["spawn_robot"]["init"]

    for add_hand, hand_side in zip([add_left_hand, add_right_hand],
                                   ["left", "right"]):

        if not add_hand:
            continue
        init_robot_state = init_state.replace(
            pos=(0.0, 0.0, -0.0),
            joint_pos=arm_joint_pos | hand_joint_pos,
        )

        config_env(object,
                   arm_actuator,
                   hand_actuator,
                   arm_action,
                   hand_action,
                   init_robot_state,
                   hand_side,
                   enabled_self_collisions=enabled_self_collisions,
                   robot_usd_file=robot_usd_files[hand_side])

    # if arm_type is not None:
    if arm_type is not None:
        if add_left_hand:
            config_reset_joint_pose(object.events,
                                    env_cfg["params"]["left_reset_joint_pose"],
                                    env_cfg["params"]["num_hand_joints"],
                                    rm_joint_limits=arm_joint_limits,
                                    hand_side="left")

        if add_right_hand:
            if env_cfg["params"].get("right_reset_joint_pose",
                                     None) is not None:

                config_reset_joint_pose(
                    object.events,
                    env_cfg["params"]["right_reset_joint_pose"],
                    env_cfg["params"]["num_hand_joints"],
                    arm_joint_limits,
                    hand_side="right")

    # if env_cfg["params"]["contact_sensor"]["init"]:
    #     if add_left_hand:
    #         config_robot_contact_sensor(object.scene, env_cfg, "left")
    #         # config_robot_contact_obs(object, env_cfg, "left")
    #     if add_right_hand:
    #         config_robot_contact_sensor(object.scene, env_cfg, "right")
    #         # config_robot_contact_obs(object, env_cfg, "right")

    if spawn_robot:

        if add_left_hand:
            spanwn_robot_hand(
                object.scene,
                env_cfg["params"]["spawn_robot"]["left_hand"]["main_path"],
                env_cfg["params"]["spawn_robot"]["spawn_hand_list"], "left")
            if env_cfg["params"]["arm_type"] is not None:
                spanwn_robot_hand(
                    object.scene,
                    env_cfg["params"]["spawn_robot"]["left_arm"]["main_path"],
                    env_cfg["params"]["spawn_robot"]["spawn_arm_list"], "left")
            config_reset_robot_pose(
                object.events,
                env_cfg,
                hand_side="left",
            )

            # config_reset_robot_setting(
            #     object.events,
            #     env_cfg,
            #     hand_side="left",
            # )

        if add_right_hand:
            spanwn_robot_hand(
                object.scene,
                env_cfg["params"]["spawn_robot"]["right_hand"]["main_path"],
                env_cfg["params"]["spawn_robot"]["spawn_hand_list"], "right")
            if env_cfg["params"]["arm_type"] is not None:

                spanwn_robot_hand(
                    object.scene,
                    env_cfg["params"]["spawn_robot"]["right_arm"]["main_path"],
                    env_cfg["params"]["spawn_robot"]["spawn_arm_list"],
                    "right")
            config_reset_robot_pose(
                object.events,
                env_cfg,
                hand_side="right",
            )
            config_reset_robot_setting(object.events,
                                       env_cfg,
                                       hand_side="right",
                                       joints_info=hand_joint_pos,
                                       arm_info=arm_joint_pos)
