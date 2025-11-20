import torch
import sys

sys.path.append(".")
import cv2

import gc
import numpy as np
import torchvision.transforms as transforms


def collect_trajectories(num_trajs,
                         env,
                         device='cpu',
                         state_name=None,
                         max_t=10,
                         interval=1,
                         camera_obs=False,
                         vis_camera=False,
                         camera_function=None,
                         resize=None):

    all_sim_params = []
    all_sim_params_classifier = []
    all_sim_params_raw_range = []
    all_sim_traj_states = []
    all_sim_traj_actions = []
    all_sim_traj_rgb = []

    sim_episode_obs = []
    sim_episode_act = []
    sim_episode_rgb = []

    observation, _ = env.reset()  # this overloaded method does reset all envs

    num_trajs_done = 0
    all_sim_params.append([])
    all_sim_params_classifier.append([])
    all_sim_params_raw_range.append([])

    while True:  # collect simulated episodes/trajectories
        actions = torch.rand(env.action_space.shape,
                             device=env.unwrapped.device) * 0
        observation, reward, terminate, time_out, info = env.step(actions)

        if time_out[0]:
            num_trajs_done += len(observation["policy"][state_name][:, 0])

            all_sim_traj_states.append(
                torch.stack(sim_episode_obs, dim=1).to("cpu"))
            all_sim_traj_actions.append(
                torch.stack(sim_episode_act, dim=1).to("cpu"))

            if camera_obs:
                all_sim_traj_rgb.append(
                    torch.stack(sim_episode_rgb, dim=1).to("cpu"))

            del sim_episode_obs
            del sim_episode_act
            del sim_episode_rgb
            gc.collect()
            torch.cuda.empty_cache()

            # Clear episode accumulators.
            sim_episode_obs = []
            sim_episode_act = []
            sim_episode_rgb = []
            print("Done", num_trajs_done)
            if num_trajs_done >= num_trajs:
                break  # collected enough trajectories (episodes)
            all_sim_params.append([])
            all_sim_params_classifier.append([])
            all_sim_params_raw_range.append([])

        else:

            sim_episode_obs.append(
                observation["policy"][state_name].to(device))
            sim_episode_act.append(env.episode_length_buf.to(device)[:, None])

            all_sim_params[-1] = observation["policy"][
                "deform_physical_params"].to("cpu")
            all_sim_params_classifier[-1] = env.scene[
                "deform_object"].parames_generator.classifer
            all_sim_params_raw_range[-1] = np.concatenate([
                env.scene["deform_object"].parames_generator.param,
                env.scene["deform_object"].parames_generator.step_param_value
            ],
                                                          axis=1)

            if camera_obs:
                rgb = camera_function(env)[:, 0]
                if resize is not None:
                    resize_transform = transforms.Resize(resize)
                    rgb = resize_transform(rgb.permute(0, 3, 1,
                                                       2)).permute(0, 2, 3, 1)

                sim_episode_rgb.append(rgb.to(device))
            if vis_camera and camera_obs:

                rgb_images = rgb.cpu()

                image = rgb_images.permute(1, 0, 2, 3).reshape(
                    rgb_images.shape[1],
                    rgb_images.shape[2] * rgb_images.shape[0], 3).numpy()
                cv2.imshow("image", image[..., ::-1])
                cv2.waitKey(2)
        if num_trajs_done >= num_trajs:
            break  # collected enough trajectories (episodes)

    sim_params_smpls = torch.cat(all_sim_params).to(device)

    sim_params_smpls_classifer = torch.as_tensor(
        np.concatenate([
            np.concatenate(all_sim_params_classifier)[..., None],
            np.concatenate(all_sim_params_raw_range)
        ],
                       axis=1)).to(device)

    sim_traj_states = torch.cat(all_sim_traj_states,
                                dim=0).to(device)[:, ::interval]
    sim_traj_actions = torch.cat(all_sim_traj_actions,
                                 dim=0).to(device)[:, ::interval]
    sim_traj_rgb = []
    if camera_obs:
        sim_traj_rgb = torch.cat(all_sim_traj_rgb,
                                 dim=0).to(device)[:, ::interval]
        sim_traj_rgb = sim_traj_rgb[..., :3]

    return sim_params_smpls, sim_traj_states, sim_traj_actions, sim_traj_rgb, sim_params_smpls_classifer
