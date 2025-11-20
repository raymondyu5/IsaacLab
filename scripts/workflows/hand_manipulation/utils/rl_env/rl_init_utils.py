import torch
from scripts.workflows.hand_manipulation.env.teleop_env.motion_planner_env import ArmMotionPlannerEnv


def init_rl_setting(object):

    if object.args_cli.add_right_hand:
        object.hand_side = "right"
    else:
        object.hand_side = "left"

    object.num_hand_joints = object.env_config["params"]["num_hand_joints"]
    object.num_arm_joints = object.env_config["params"]["num_arm_joints"]
    object.target_manipulated_object = object.env_config["params"][
        "target_manipulated_object"]

    arm_action_bound = torch.as_tensor(
        object.env_config["params"]["Task"]["action_range"]).to(object.device)

    arm_action_limit = torch.stack([
        torch.tensor([-arm_action_bound[0]] * 3 + [-arm_action_bound[1]] * 3 +
                     [-arm_action_bound[2]] * object.num_hand_joints,
                     device=object.device),
        torch.tensor([arm_action_bound[0]] * 3 + [arm_action_bound[1]] * 3 +
                     [arm_action_bound[2]] * object.num_hand_joints,
                     device=object.device)
    ],
                                   dim=1)
    object.lower_bound = arm_action_limit[:, 0]
    object.upper_bound = arm_action_limit[:, 1]

    init_ee_pose = torch.as_tensor(
        object.env_config["params"]["init_ee_pose"]).to(
            object.device).unsqueeze(0)

    object.arm_motion_env = ArmMotionPlannerEnv(
        object.env,
        object.args_cli,
        object.env_config,
    )

    init_arm_qpos = object.arm_motion_env.ik_plan_motion(
        init_ee_pose).repeat_interleave(object.env.num_envs, dim=0)
    init_hand_qpos = torch.zeros(
        (object.env.num_envs, object.num_hand_joints)).to(object.device)
    object.init_robot_qpos = torch.cat([init_arm_qpos, init_hand_qpos],
                                       dim=1).to(object.device)

    object.env.scene[
        f"{object.hand_side}_hand"].data.reset_joint_pos = object.init_robot_qpos
    hand_joint_limits = object.env.scene[
        f"{object.hand_side}_hand"].data.joint_limits[0,
                                                      -object.num_hand_joints:]

    object.horizon = object.env_config["params"]["Task"]["horizon"]

    init_arm_qpos = object.arm_motion_env.ik_plan_motion(
        init_ee_pose).repeat_interleave(object.env.num_envs, dim=0)
    init_hand_qpos = torch.zeros(
        (object.env.num_envs, object.num_hand_joints)).to(object.device)
    object.init_robot_qpos = torch.cat([init_arm_qpos, init_hand_qpos],
                                       dim=1).to(object.device)
    object.env.scene[
        f"{object.hand_side}_hand"].data.reset_joint_pos = object.init_robot_qpos
