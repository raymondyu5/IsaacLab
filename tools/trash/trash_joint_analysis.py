import h5py
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

normalize_file = h5py.File(
    "logs/1113_placement_tabletop_bbox/grasp_normalized_noise_aug.hdf5", 'r+')

joint_pos_buffer = []
episode_lengths = []
robot_base_buffer = []
for i in range(len(normalize_file["data"])):
    joint_pos_buffer.append(
        normalize_file["data"][f"demo_{i}"]["obs"]["ee_pose"])
    # joint_pos_buffer.append(normalize_file["data"][f"demo_{i}"]["actions"])

    episode_lengths.append(  # Store the length of each episode
        normalize_file["data"][f"demo_{i}"]["obs"]["ee_pose"].shape[0])
    robot_base_buffer.append(
        normalize_file["data"][f"demo_{i}"]["obs"]["robot_base"][0][:3])

import numpy as np

robot_base_buffer = np.concatenate(robot_base_buffer, axis=0).reshape(-1, 3)
joint_pos_buffer = np.concatenate(joint_pos_buffer, axis=0)[..., :8]

# Extract x, y, z coordinates
x = robot_base_buffer[:, 0]
y = robot_base_buffer[:, 1]
z = robot_base_buffer[:, 2]

# Create a 3D plot
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Scatter plot of the 3D points
# ax.scatter(x, y, z, c='b', marker='o', alpha=0.6, s=100)

# # Set labels for each axis
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# ax.set_title("3D Distribution of Robot Base Positions")

# # Show plot
# plt.show()

joint_pos_buffer[:, -1] = np.clip(joint_pos_buffer[:, -1], -1, 1)

fig, axes = plt.subplots(2, 4, figsize=(15, 9))
axes = axes.flatten()

start_idx = 0

for ep, ep_length in enumerate(episode_lengths):
    end_idx = start_idx + ep_length  # Define the end index for the episode

    # Plot each joint's data for this episode on the corresponding subplot
    for i in range(8):  # Loop over each joint
        axes[i].plot(range(ep_length),
                     joint_pos_buffer[start_idx:end_idx, i],
                     label=f"Episode {ep + 1}")
        axes[i].set_title(f"Joint {i + 1} Position")
        axes[i].set_xlabel("Time Step")
        axes[i].set_ylabel("Position")

    # Update start index for the next episode
    start_idx = end_idx

# Adjust layout and show plot
plt.tight_layout()
plt.legend(loc="upper right", fontsize="small")
plt.show()
