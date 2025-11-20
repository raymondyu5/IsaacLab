import matplotlib.pyplot as plt
import torchvision
import torch
import os

import numpy as np

import torchvision.transforms.functional as TF
# import cv2
#
import kaolin.metrics.pointcloud as pc


def save_target_video(images_buffer,
                      log_path,
                      id,
                      folder_name="video",
                      num_explore_actions=3):

    images_buffer = torch.stack(images_buffer)[..., :3]

    images_buffer = images_buffer.permute(0, 1, 3, 4, 2, 5)

    num_images = images_buffer.shape[4]
    concatenated_images = torch.cat(
        [images_buffer[:, :, :, :, i, :] for i in range(num_images)], dim=3)

    os.makedirs(log_path + "/" + folder_name, exist_ok=True)

    concatenated_images_horizontal = torch.cat(
        [concatenated_images[:, i] for i in range(num_explore_actions)], dim=2)

    torchvision.io.write_video(log_path + f"/{folder_name}/loop_{id}.mp4",
                               concatenated_images_horizontal.cpu(),
                               fps=10)
    del concatenated_images_horizontal, concatenated_images, images_buffer, num_images
    torch.cuda.empty_cache()


def collect_pre_env_trajectories(env,
                                 buffer,
                                 delta_pose,
                                 gripper_offset_xyz,
                                 cache_type,
                                 log_path=None,
                                 id=None,
                                 image_key="rgb"):
    from tools.deformable_obs import object_3d_observation

    if env.episode_length_buf[0] == 0:
        last_obs = env.reset()[0]

    if cache_type == "target":
        images_buffer = []

    for _ in range(env.max_episode_length):
        transition = {}
        abs_pos = delta_pose[:, :3] * (
            env.episode_length_buf[:, None].repeat_interleave(3, 1)) * 1.5
        abs_quat = delta_pose[:, 3:7]
        actions = torch.cat([abs_pos, abs_quat, delta_pose[:, -1][..., None]],
                            dim=1)
        actions[:, :3] += gripper_offset_xyz

        next_obs, reward, terminate, time_out, info = env.step(actions)

        transition["next_obs"] = next_obs["policy"]
        transition["obs"] = last_obs["policy"]
        transition["reward"] = reward * 0
        transition["action"] = actions
        if cache_type == "target":

            images_buffer.append(transition["obs"][image_key])

        buffer.cache_traj(transition, cache_type=cache_type)

        last_obs = next_obs

        del next_obs

    if cache_type == "target":
        save_target_video(images_buffer, log_path, id)
    buffer.store_transition(cache_type)
    buffer.clear_cache(cache_type)


def collect_env_trajectories(env,
                             buffer,
                             delta_pose,
                             gripper_offset_xyz,
                             cache_type,
                             log_path=None,
                             id=None,
                             image_key="rgb"):
    from tools.deformable_obs import object_3d_observation

    if env.episode_length_buf[0] == 0:
        last_obs = env.reset()[0]

    if cache_type == "target":
        images_buffer = []

    for _ in range(env.max_episode_length):
        transition = {}
        abs_pos = delta_pose[:, :3] * (
            env.episode_length_buf[:, None].repeat_interleave(3, 1)) * 1.5
        abs_quat = delta_pose[:, 3:7]
        actions = torch.cat([abs_pos, abs_quat, delta_pose[:, -1][..., None]],
                            dim=1)
        actions[:, :3] += gripper_offset_xyz

        next_obs, reward, terminate, time_out, info = env.step(actions)

        transition["next_obs"] = next_obs["policy"]
        transition["obs"] = last_obs["policy"]
        transition["reward"] = reward * 0
        transition["action"] = actions
        if cache_type == "target":

            images_buffer.append(transition["obs"][image_key])

        buffer.cache_traj(transition, cache_type=cache_type)

        last_obs = next_obs

        del next_obs

    if cache_type == "target":
        save_target_video(images_buffer, log_path, id)
    buffer.store_transition(cache_type)
    buffer.clear_cache(cache_type)


from mpl_toolkits.mplot3d import Axes3D


def plot_3d_result(log_path):

    num_files = len(os.listdir(log_path + "/result"))

    target_propoerties = []
    rollout_properties = []
    rollout_properties_std = []
    for i in range(num_files):

        params = np.load(log_path + f"/result/{i}.npz")["result"]

        target_propoerties.append(params[:3])
        rollout_properties.append(params[3:6])
        rollout_properties_std.append(params[6:9])
    params_name = np.load(log_path + f"/result/{i}.npz")["param_names"]

    target_properties = np.array(target_propoerties)
    rollout_properties = np.array(rollout_properties)
    rollout_properties_std = np.array(rollout_properties_std)
    save_plot_to_txt(log_path, target_properties, rollout_properties,
                     rollout_properties_std)

    # plot 1s\
    for i, name_a in enumerate(params_name):
        plt.scatter(target_properties[:, i],
                    target_properties[:, i],
                    label='Target Properties',
                    color='blue',
                    s=60)

        plt.plot(rollout_properties[:, i],
                 rollout_properties[:, i],
                 label='Rollout Properties',
                 color='red')

        plt.scatter(rollout_properties[:, i],
                    rollout_properties[:, i],
                    label='Rollout Properties',
                    color='red')

        # Plot predicted properties with error bars (red line with error bars)
        plt.errorbar(
            rollout_properties[:, i],
            rollout_properties[:, i],
            xerr=abs(rollout_properties_std[:, i]),
            label='std',
            color='red',
            fmt='o',  # Marker for points
            capsize=5)  # Add caps to the error bars

        plt.ylabel(name_a)
        plt.legend()

        plt.savefig(f"{log_path}/{name_a}.png")

        plt.cla()

    # plot 2d
    for i, name_a in enumerate(params_name):
        for j, name_b in enumerate(params_name):
            if i >= j:
                continue
            plt.figure()
            plt.scatter(target_properties[:, i],
                        target_properties[:, j],
                        label='Target Properties',
                        color='blue')
            plt.plot(target_properties[:, i],
                     target_properties[:, j],
                     label='Target Properties',
                     color='blue')
            plt.scatter(rollout_properties[:, i],
                        rollout_properties[:, j],
                        label='Rollout Properties',
                        color='red')
            plt.plot(rollout_properties[:, i],
                     rollout_properties[:, j],
                     label='Rollout Properties',
                     color='red')
            # plt.errorbar(rollout_properties[:, i],
            #              rollout_properties[:, j],
            #              xerr=abs(rollout_properties_std[:, i]),
            #              yerr=abs(rollout_properties_std[:, j]),
            #              label='std',
            #              color='red',
            #              fmt='o',
            #              capsize=5)
            plt.xlabel(params_name[i])
            plt.ylabel(params_name[j])
            plt.legend()
            plt.savefig(f"{log_path}/{params_name[i]}_{params_name[j]}.png")
            plt.cla()
    # Convert lists to numpy arrays for plotting
    target_properties_np = np.array(target_properties)
    rollout_properties_np = np.array(rollout_properties)
    rollout_properties_std_np = np.array(rollout_properties_std)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot target properties (blue line)
    ax.plot(target_properties_np[:, 0],
            target_properties_np[:, 1],
            target_properties_np[:, 2],
            label='Target Properties',
            color='blue')

    # Plot target properties (blue line)
    ax.scatter(target_properties_np[:, 0],
               target_properties_np[:, 1],
               target_properties_np[:, 2],
               label='Target Properties',
               color='blue')

    # Plot rollout properties (red line)
    ax.scatter(rollout_properties_np[:, 0],
               rollout_properties_np[:, 1],
               rollout_properties_np[:, 2],
               label='Rollout Properties',
               color='red',
               s=60)

    ax.plot(
        rollout_properties_np[:, 0],
        rollout_properties_np[:, 1],
        rollout_properties_np[:, 2],
        label='Rollout Properties',
        color='red',
    )

    # # Plot predicted properties with error bars (red line with error bars)
    # ax.errorbar(rollout_properties_np[:, 0],
    #             rollout_properties_np[:, 1],
    #             rollout_properties_np[:, 2],
    #             xerr=abs(rollout_properties_std_np[:, 0]),
    #             yerr=abs(rollout_properties_std_np[:, 1]),
    #             zerr=abs(rollout_properties_std_np[:, 2]),
    #             label='std',
    #             fmt='o',
    #             color='red',
    #             capsize=5)  # Add caps to the error bars

    # Set axis labels
    ax.set_xlabel(params_name[0])
    ax.set_ylabel(params_name[1])
    ax.set_zlabel(params_name[2])

    # Add legend
    ax.legend()

    # Save the plot
    plt.savefig(
        f"{log_path}/{params_name[0]}_{params_name[1]}_{params_name[2]}.png")
    # plt.show()
    # Clear the figure
    plt.cla()
    plt.clf()
    plt.close()


def plot_1d_result(log_path):

    num_files = len(os.listdir(log_path + "/result"))

    target_propoerties = []
    rollout_properties = []
    rollout_properties_std = []

    if num_files < 5:
        return None
    for i in range(num_files):

        params = np.load(log_path + f"/result/{i}.npz")["result"]

        target_propoerties.append(params[:1])
        rollout_properties.append(params[1:2])
        rollout_properties_std.append(params[2:3])
    params_name = np.load(log_path + f"/result/{i}.npz")["param_names"]

    # Convert lists to numpy arrays for plotting
    target_properties_np = np.array(target_propoerties)
    rollout_properties_np = np.array(rollout_properties)
    rollout_properties_std_np = np.array(rollout_properties_std)
    save_plot_to_txt(log_path, target_properties_np, rollout_properties_np,
                     rollout_properties_std_np)

    plt.scatter(target_properties_np[:, 0],
                target_properties_np[:, 0],
                label='Target Properties',
                color='blue')

    plt.plot(rollout_properties_np[:, 0],
             rollout_properties_np[:, 0],
             label='Rollout Properties',
             color='red')

    plt.scatter(rollout_properties_np[:, 0],
                rollout_properties_np[:, 0],
                label='Rollout Properties',
                color='red')

    # Plot predicted properties with error bars (red line with error bars)
    plt.errorbar(
        rollout_properties_np[:, 0],
        rollout_properties_np[:, 0],
        xerr=abs(rollout_properties_std_np[:, 0]),
        label='std',
        color='red',
        fmt='o',  # Marker for points
        capsize=5)  # Add caps to the error bars

    plt.ylabel(params_name[0])
    plt.legend()

    plt.savefig(f"{log_path}/{params_name[0]}.png")

    plt.cla()


def save_plot_to_txt(log_path, target_properties_np, rollout_properties_np,
                     rollout_properties_std_np) -> None:

    # save text file
    file_name = "params_output.txt"
    # Save the data in the desired format
    with open(log_path + f"/{file_name}", 'w') as f:
        for i in range(
                target_properties_np.shape[0]):  # Iterate over the 64 entries
            # Write Real data
            f.write(f"Real: {i}, mean: {target_properties_np[i].tolist()}\n")

            # Write Rollout data
            f.write(
                f"Rollout: {i}, mean: {rollout_properties_np[i].tolist()}, std: {rollout_properties_std_np[i].tolist()}\n"
            )
            f.write("=========================================\n")


def plot_2d_visualization(target_properties_np, rollout_properties_np,
                          rollout_properties_std_np, params_name, log_path,
                          png_name):

    plt.scatter(target_properties_np[:, 0],
                target_properties_np[:, 1],
                label='Target Properties',
                color='blue',
                s=40)

    plt.plot(
        target_properties_np[:, 0],
        target_properties_np[:, 1],
        label='Target Properties',
        color='blue',
    )

    plt.plot(rollout_properties_np[:, 0],
             rollout_properties_np[:, 1],
             label='Rollout Properties',
             color='red')

    plt.scatter(rollout_properties_np[:, 0],
                rollout_properties_np[:, 1],
                label='Rollout Properties',
                color='red',
                s=40)

    # Plot predicted properties with error bars (red line with error bars)

    plt.errorbar(
        rollout_properties_np[:, 0],
        rollout_properties_np[:, 1],
        xerr=abs(rollout_properties_std_np[:, 0]),
        yerr=abs(rollout_properties_std_np[:, 1]),
        label='std',
        color='red',
        fmt='o',  # Marker for points
        capsize=5)  # Add caps to the error bars

    # Add labels and legend
    plt.xlabel(params_name[0])
    plt.ylabel(params_name[1])
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.savefig(f"{log_path}/{png_name}.png")
    plt.clf()
    plt.cla()


import imageio


# Function to add text to an image
def add_text(image,
             text,
             position=(50, 50),
             font_scale=1,
             font_color=(0, 0, 255),
             thickness=2,
             line_height=100):
    import cv2
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = position

    # Split text into lines by newline characters
    for i, line in enumerate(text.split('\n')):
        cv2.putText(image, line, (x, y + i * line_height), font, font_scale,
                    font_color, thickness)


# Function to handle both 1D and 2D videos
def process_video(log_path, dimensions=2):
    # Load parameters from result files
    num_files = len(os.listdir(log_path + "/result"))
    import cv2

    target_properties = []
    rollout_properties = []
    if num_files < 5:
        return None
    for i in range(num_files):
        params = np.load(log_path + f"/result/{i}.npz")["result"]
        if dimensions == 1:
            target_properties.append(params[:1])  # Only 1 parameter
            rollout_properties.append(params[1:2])
        else:
            target_properties.append(params[:2])  # Two parameters
            rollout_properties.append(params[2:4])

    params_name = np.load(log_path + f"/result/{i}.npz")["param_names"]

    # Define directories
    eval_dir = f"{log_path}/eval/video"
    target_dir = f"{log_path}/target/video"
    output_dir = f"{log_path}/output/video"

    # Create output directory if not exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop through videos in the eval directory
    for filename in os.listdir(eval_dir):
        if filename.endswith(('.mp4', '.avi')):
            eval_video_path = os.path.join(eval_dir, filename)
            target_video_path = os.path.join(target_dir, filename)

            # Check if corresponding video exists in the target directory
            if os.path.isfile(target_video_path):
                output_file = os.path.join(output_dir, f"combined_{filename}")

                # Open both videos
                eval_cap = cv2.VideoCapture(eval_video_path)
                target_cap = cv2.VideoCapture(target_video_path)

                # Get properties from eval video
                fps = eval_cap.get(cv2.CAP_PROP_FPS)
                width = int(eval_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(eval_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # Create a VideoWriter object for the output video
                out = cv2.VideoWriter(output_file,
                                      cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                      (width * 2, height))

                i = 0  # Frame index
                while eval_cap.isOpened() and target_cap.isOpened():
                    ret_eval, eval_frame = eval_cap.read()
                    ret_target, target_frame = target_cap.read()

                    # Break the loop if either video ends
                    if not ret_eval or not ret_target:
                        break

                    # Resize target frame to match eval frame size if necessary
                    target_frame = cv2.resize(target_frame, (width, height))

                    # Concatenate the two frames horizontally
                    combined_frame = np.hstack((target_frame, eval_frame))

                    # Generate text depending on dimensions (1D or 2D)

                    params_index = int(filename.split(".")[0].split("_")[1])
                    if dimensions == 1:
                        text = (
                            f"Real:{params_name[0]}: {target_properties[params_index][0]:.2f},\n"
                            f"Rollout:{params_name[0]}: {rollout_properties[params_index][0]:.2f}"
                        )
                    else:
                        text = (
                            f"Real:{params_name[0]}: {target_properties[params_index][0]:.2f}, {params_name[1]}: {target_properties[params_index][1]:.2f},\n"
                            f"Rollout:{params_name[0]}: {rollout_properties[params_index][0]:.2f}, {params_name[1]}: {rollout_properties[params_index][1]:.2f}"
                        )

                    # Add red text to the combined frame
                    add_text(combined_frame,
                             text,
                             position=(int(combined_frame.shape[1] / 3), 100),
                             font_color=(0, 0, 255),
                             font_scale=1.0,
                             thickness=2)

                    # Write the frame to the output video
                    out.write(combined_frame)

                    i += 1

                # Release video capture and writer
                eval_cap.release()
                target_cap.release()
                out.release()

                print(
                    f"Combined {filename} from eval and target into {output_file}"
                )
            else:
                print(f"Matching video for {filename} not found in target.")


# Function to handle 1D video (with one parameter)
def make_1d_video(log_path):
    process_video(log_path, dimensions=1)


# Function to handle 2D video (with two parameters)
def make_2d_video(log_path):
    process_video(log_path, dimensions=2)


def plot_2d_result(log_path, min_batch=10):

    num_files = len(os.listdir(log_path + "/result"))

    target_propoerties = []
    rollout_properties = []
    rollout_properties_std = []
    for i in range(num_files):

        params = np.load(log_path + f"/result/{i}.npz")["result"]

        target_propoerties.append(params[:2])
        rollout_properties.append(params[2:4])
        rollout_properties_std.append(params[4:6])
    params_name = np.load(log_path + f"/result/{i}.npz")["param_names"]

    # Convert lists to numpy arrays for plotting
    target_properties_np = np.array(target_propoerties)
    rollout_properties_np = np.array(rollout_properties)
    rollout_properties_std_np = np.array(rollout_properties_std)
    save_plot_to_txt(log_path, target_properties_np, rollout_properties_np,
                     rollout_properties_std_np)
    plot_2d_visualization(target_properties_np, rollout_properties_np,
                          rollout_properties_std_np, params_name, log_path,
                          f"{params_name[0]}_{params_name[1]}")

    num_batch = int(len(target_properties_np) / min_batch)
    for i in range(num_batch):
        os.makedirs(log_path + f"/result_images/", exist_ok=True)
        start_frame = i * min_batch
        end_frame = (i + 1) * min_batch
        plot_2d_visualization(
            target_properties_np[start_frame:end_frame],
            rollout_properties_np[start_frame:end_frame],
            rollout_properties_std_np[start_frame:end_frame], params_name,
            log_path, f"/result_images/{params_name[0]}_{params_name[1]}_{i}")


def evaluate_chamfer_distance(target_pc_seq, train_pc_seq, min_batch_size=32):

    batch, num_exploration_actions, seq, num_pc, dim = train_pc_seq.shape

    batch_loss = torch.zeros(batch)

    # Process in mini-batches
    for start in range(0, batch, min_batch_size):
        end = min(start + min_batch_size, batch)
        batch_length = end - start

        # Select mini-batch
        train_minibatch = train_pc_seq[start:end, ...]

        target_minibatch = target_pc_seq[None].repeat_interleave(
            batch_length, 0)

        train_pc_seq_reshape = train_minibatch.contiguous().view(
            batch_length * num_exploration_actions * seq, num_pc, dim)

        target_pc_seq_reshape = target_minibatch.contiguous().view(
            batch_length * num_exploration_actions * seq, num_pc, dim)

        loss = pc.chamfer_distance(train_pc_seq_reshape, target_pc_seq_reshape)
        loss = loss.contiguous().view(batch_length, num_exploration_actions,
                                      seq)

        batch_loss[start:end] = torch.sum(loss.reshape(batch_length, -1),
                                          dim=1)

    return 1 / batch_loss.cpu().numpy(), batch_loss
