from scripts.workflows.open_policy.utils.criterion import *

import torch
import isaaclab.utils.math as math_utils


def get_root_link_state(env, object_name):
    object = env.scene[object_name]

    return object.data.root_link_state_w.clone()


def add_ee_offset(env, ee_pos_w):
    ee_pose_b, ee_quat_b = math_utils.combine_frame_transforms(
        ee_pos_w[:, :3], ee_pos_w[:, 3:7],
        torch.tensor([[0.0000, 0.0000, 0.1070]],
                     device=env.device).repeat_interleave(env.num_envs, 0),
        torch.tensor([[1., 0., 0., 0.]],
                     device=env.device).repeat_interleave(env.num_envs, 0))
    ee_pos_w = torch.cat([ee_pose_b, ee_quat_b], dim=1)
    return ee_pos_w


def criterion_pick_place(env,
                         pick_object_name,
                         place_object_name,
                         bbox_region,
                         args_cli=None):

    pick_object_state = get_root_link_state(env, pick_object_name)
    placment_object_state = get_root_link_state(env, place_object_name)
    if args_cli is not None:
        if args_cli.pick_only:
            pick_or_not = pick_object_state[:, 2] > 0.10
            success_rate = pick_or_not.sum().float() / pick_or_not.shape[0]
            return pick_or_not, success_rate.item()
        if args_cli.approach_only:

            ee_pose = get_root_link_state(env, "panda_hand")
            ee_pose = add_ee_offset(env, ee_pose)
            delta_xyz_dist = torch.linalg.norm(ee_pose[:, :3] -
                                               pick_object_state[:, :3],
                                               dim=1)

            approach_or_not = delta_xyz_dist < 0.02
            success_rate = approach_or_not.sum().float(
            ) / approach_or_not.shape[0]

            return approach_or_not, success_rate.item()
    delta_xyz_dist = abs(pick_object_state[:, :3] -
                         placment_object_state[:, :3])
    place_or_not = (delta_xyz_dist[:, 0] < bbox_region[0]) & (
        delta_xyz_dist[:, 1] < bbox_region[1]) & (pick_object_state[:, 2]
                                                  < bbox_region[2])
    success_rate = place_or_not.sum().float() / pick_object_state.shape[0]
    return place_or_not, success_rate.item()


def vis_obs(obs):
    # Create a figure and subplots
    plt.figure(figsize=(12, 8))
    obs = obs.cpu().numpy()

    for i in range(obs.shape[1]):
        plt.plot(obs[:, i], label=f'Plot {i+1}')

    # Add labels, legend, and title
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Visualization of N Plots')
    plt.legend()
    plt.show()
