# import matplotlib.pyplot as plt
# import numpy as np
# import h5py
# import torch
# import isaaclab.utils.math as math_utils
# # Load the grasp data
# normalized_grasp = h5py.File(
#     "/home/ensu/Documents/weird/IsaacLab/logs/test/delta_teleop_data.hdf5",
#     'r+')
# data = []

# # Extract the first 10 demos' action trajectories (assuming action data has 3D coordinates)
# for i in range(200):  # Previous 10 demos
#     action = np.array(normalized_grasp["data"][f"demo_{i}"]["actions"][:, :7])
#     # robot_base = torch.as_tensor(normalized_grasp["data"][f"demo_{i}"]["obs"]["robot_base"][:, :7]
#     data.append(action)

# # Plot trajectories for each demo
# fig, ax = plt.subplots()
# for i, action in enumerate(data):
#     ax.plot(action[:, 0], action[:, 1], label=f'Demo {i+1}')

# ax.set_xlabel('X Position')
# ax.set_ylabel('Y Position')
# ax.set_title('Trajectories of the Previous 10 Demos')
# ax.legend()
# plt.show()

import matplotlib.pyplot as plt
import numpy as np
import h5py

# Load the grasp data
normalized_grasp = h5py.File("logs/teleop_0528/normalized_mustard_bottle.hdf5",
                             'r+')

data = []
rewards = []
# Extract the first 200 demos' action trajectories (first 3 dims: x, y, z)
for i in range(20):
    action = np.array(normalized_grasp["data"][f"demo_{i}"]["actions"]
                      [:, :3])  # Only x, y, z
    rewards.append(
        np.array(normalized_grasp["data"][f"demo_{i}"]["rewards"][:]))
    data.append(action)

# Plot the x, y, z values over time
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

for demo_id, action in enumerate(data):
    timesteps = np.arange(action.shape[0])
    axes[0].plot(timesteps, action[:, 0])  # X
    axes[1].plot(timesteps, action[:, 1])  # Y
    axes[2].plot(timesteps, action[:, 2])  # Z

axes[0].set_ylabel('X')
axes[1].set_ylabel('Y')
axes[2].set_ylabel('Z')
axes[2].set_xlabel('Timestep')

axes[0].set_title('First 3 Action Dimensions Over Time (X, Y, Z)')
axes[0].legend(fontsize='small', ncol=4,
               loc='upper right')  # optional, remove if too cluttered

plt.tight_layout()
plt.show()

# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for demo_id, action in enumerate(data):
    ax.plot(action[:, 0], action[:, 1], action[:, 2], alpha=0.6)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Trajectories of Action (X, Y, Z) for 200 Demos')

plt.tight_layout()
plt.show()
