import torch
import isaaclab.utils.math as math_utils


def reset_root_state_uniform(env,
                             pose_range,
                             root_states,
                             asset_name,
                             env_ids,
                             scale_factor=1.0,
                             reset_root_state=True):

    # extract the used quantities (to enable type-hinting)

    asset = env.scene[asset_name]

    root_velocity = asset.data.default_root_state[env_ids].clone()

    # poses
    range_list = [
        pose_range.get(key, (0.0, 0.0))
        for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]

    ranges = torch.tensor(range_list, device=asset.device)
    ranges[3:] *= scale_factor
    rand_samples = math_utils.sample_uniform(ranges[:, 0],
                                             ranges[:, 1], (len(env_ids), 6),
                                             device=asset.device)

    positions = root_states[:, 0:3] + env.scene.env_origins[
        env_ids] + rand_samples[:, 0:3]

    orientations_delta = math_utils.quat_from_euler_xyz(
        rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(orientations_delta, root_states[:, 3:7])
    # velocities
    range_list = [(0.0, 0.0)
                  for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0],
                                             ranges[:, 1], (len(env_ids), 6),
                                             device=asset.device)
    if reset_root_state:
        velocities = root_velocity[:, 7:13] + rand_samples
        asset.data.reset_root_state = torch.cat(
            [positions, orientations, velocities], dim=-1)

        # set into the physics simulation
        asset.write_root_pose_to_sim(torch.cat([positions, orientations],
                                               dim=-1),
                                     env_ids=env_ids)
        asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)

    return positions, orientations
