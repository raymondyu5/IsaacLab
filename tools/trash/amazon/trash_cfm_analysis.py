# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# from collections import defaultdict

# result = np.load(
#     "logs/data_0705/retarget_visionpro_data/rl_data/eval_results/bunny/analysis.npy",
#     allow_pickle=True)
# result_dict = {tuple(item[0]): float(item[1]) for item in result}
# import matplotlib.pyplot as plt
# import numpy as np
# from collections import defaultdict

# rounded_dict = {
#     tuple(round(v, 2) for v in k): val
#     for k, val in result_dict.items()
# }

# # group by theta
# theta_groups = defaultdict(list)
# for (x, y, theta), val in rounded_dict.items():
#     # theta = (theta % (np.pi) * 180)
#     theta_groups[theta].append((x, y, val))

# # plot
# n_theta = len(theta_groups)
# ncols = min(3, n_theta)  # up to 3 subplots per row
# nrows = int(np.ceil(n_theta / ncols))

# fig, axes = plt.subplots(nrows,
#                          ncols,
#                          figsize=(5 * ncols, 4 * nrows),
#                          squeeze=False,
#                          constrained_layout=True)

# all_images = []
# for ax, (theta, entries) in zip(axes.flat, sorted(theta_groups.items())):
#     xs = sorted(set(x for x, _, _ in entries))
#     ys = sorted(set(y for _, y, _ in entries))

#     # build grid
#     grid = np.full((len(ys), len(xs)), np.nan)
#     for x, y, val in entries:
#         xi, yi = xs.index(x), ys.index(y)
#         grid[yi, xi] = val

#     im = ax.imshow(grid,
#                    origin="lower",
#                    extent=[min(xs), max(xs),
#                            min(ys), max(ys)],
#                    vmin=0,
#                    vmax=1,
#                    cmap="viridis")
#     ax.set_title(f"Theta = {theta:.2f}")
#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     all_images.append(im)

# # Add a single top colorbar
# cbar = fig.colorbar(all_images[0],
#                     ax=axes,
#                     orientation="horizontal",
#                     shrink=0.7,
#                     pad=0.1)
# cbar.set_label("Success Rate")

# plt.suptitle("2D Success Rate Maps per Theta", fontsize=16, y=1.02)
# plt.show()
####--------- second part: make a video grid from multiple zarr files -----------####
import os
import zarr
import numpy as np
import cv2
import imageio
import re

load_path = "logs/data_0705/retarget_visionpro_data/rl_data/analysis"

# Collect all zarr files
zarr_files = sorted(
    [f for f in os.listdir(load_path) if f.endswith(".zarr")],
    key=lambda x: int(re.search(r"\d+", x).group())  # extract number
)

zarr_files = zarr_files[-36:]  # take first 36 files

# Parameters
resize_shape = (224, 224)  # (h, w)
grid_size = (6, 6)

# Load all sequences into memory (list of arrays, each [T, H, W, 3])
sequences = []

for file in zarr_files:
    data = zarr.open(os.path.join(load_path, file), mode="r")
    rgb = np.array(data["data/rgb_0"])  # shape [T, H, W, 3]

    obs_keys = []
    data.visit(lambda k: obs_keys.append(k)
               if k.startswith("data/") and k != "data" else None)
    sequences.append(rgb)

# Find shortest length among sequences
min_len = min(seq.shape[0] for seq in sequences)
print(f"Each grid will have {min_len} frames.")

# Open video writer
save_path = os.path.join(load_path, "grid_video.mp4")
writer = imageio.get_writer(save_path, fps=10, codec="libx264")

# Loop over timesteps
for t in range(min_len):
    images = []
    for seq in sequences:
        frame = seq[t]  # take t-th frame
        img_resized = cv2.resize(frame.astype(np.uint8),
                                 resize_shape[::-1])  # cv2 needs (w,h)
        images.append(img_resized)

    # Create canvas for 6x6 grid
    grid_h, grid_w = grid_size[0] * resize_shape[0], grid_size[
        1] * resize_shape[1]
    canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    for idx, img in enumerate(images):
        row = idx // grid_size[1]
        col = idx % grid_size[1]
        y, x = row * resize_shape[0], col * resize_shape[1]
        canvas[y:y + resize_shape[0], x:x + resize_shape[1]] = img

    # Add frame to video
    writer.append_data(canvas)

writer.close()
print(f"Saved video at {save_path}")

# ## save action sequences
# import os
# import zarr
# import numpy as np
# import matplotlib.pyplot as plt

# load_path = "logs/data_0705/retarget_visionpro_data/rl_data/analysis"

# # Storage
# actions = []
# ee_poses = []

# for file in zarr_files:
#     data = zarr.open(os.path.join(load_path, file), mode="r")
#     actions.append(np.array(data["data/actions"])[..., :3])  # (T, 3)
#     ee_poses.append(np.array(data["data/right_ee_pose"])[..., :3])  # (T, 3)

# # Compute global y-limits
# all_actions = np.concatenate(actions, axis=0)  # (N, 3)
# all_ee = np.concatenate(ee_poses, axis=0)  # (N, 3)

# act_min, act_max = all_actions.min(), all_actions.max()
# ee_min, ee_max = all_ee.min(), all_ee.max()

# print(f"Action range: {act_min:.3f} ~ {act_max:.3f}")
# print(f"EE pose range: {ee_min:.3f} ~ {ee_max:.3f}")

# # === Plot EE pose ===
# fig, axes = plt.subplots(6, 6, figsize=(24, 24))
# for idx, ax in enumerate(axes.flatten()):
#     if idx >= len(ee_poses):
#         ax.axis("off")
#         continue
#     timesteps = np.arange(ee_poses[idx].shape[0])
#     ax.plot(timesteps, ee_poses[idx][:, 0], label="ee_x")
#     ax.plot(timesteps, ee_poses[idx][:, 1], label="ee_y")
#     ax.plot(timesteps, ee_poses[idx][:, 2], label="ee_z")
#     ax.set_title(f"File {idx+1}", fontsize=10)
#     ax.set_xlabel("t")
#     ax.set_ylabel("pose")
#     ax.set_ylim(ee_min, ee_max)  # fixed y-limits
#     if idx == 0:
#         ax.legend(fontsize=8)
# plt.tight_layout()
# plt.savefig(os.path.join(load_path, "ee_pose_grid.png"))
# plt.close()

# # === Plot Actions ===
# fig, axes = plt.subplots(6, 6, figsize=(24, 24))
# for idx, ax in enumerate(axes.flatten()):
#     if idx >= len(actions):
#         ax.axis("off")
#         continue
#     timesteps = np.arange(actions[idx].shape[0])
#     ax.plot(timesteps, actions[idx][:, 0], label="act_x")
#     ax.plot(timesteps, actions[idx][:, 1], label="act_y")
#     ax.plot(timesteps, actions[idx][:, 2], label="act_z")
#     ax.set_title(f"File {idx+1}", fontsize=10)
#     ax.set_xlabel("t")
#     ax.set_ylabel("action")
#     ax.set_ylim(act_min, act_max)  # fixed y-limits
#     if idx == 0:
#         ax.legend(fontsize=8)
# plt.tight_layout()
# plt.savefig(os.path.join(load_path, "actions_grid.png"))
# plt.close()

# print("Saved ee_pose_grid.png and actions_grid.png with fixed y-limits.")

import os
import zarr
import numpy as np
import matplotlib.pyplot as plt

load_path = "logs/data_0705/retarget_visionpro_data/rl_data/analysis"

ee_poses = []
for file in zarr_files:
    data = zarr.open(os.path.join(load_path, file), mode="r")
    ee = np.array(data["data/right_ee_pose"])[..., :3]  # (T, 3)
    ee_poses.append(ee)

# Align to shortest rollout
min_len = min(ee.shape[0] for ee in ee_poses)
ee_stack = np.stack([ee[:min_len] for ee in ee_poses], axis=0)  # (N, T, 3)

timesteps = np.arange(min_len)

# === One figure with 3 subplots (trajectories + mean ± std) ===
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

labels = ["ee_x(meters)", "ee_y(meters)", "ee_z(meters)"]
titles = [
    "All EE X trajectories", "All EE Y trajectories", "All EE Z trajectories"
]

for i in range(3):
    # Plot all trajectories
    for ee in ee_poses:
        axes[i].plot(np.arange(ee.shape[0]), ee[:, i], alpha=0.3, color="gray")

    # Mean & std
    mean = ee_stack[:, :, i].mean(axis=0)
    std = ee_stack[:, :, i].std(axis=0)
    axes[i].plot(timesteps, mean, color="blue", linewidth=2, label="mean")
    axes[i].fill_between(timesteps,
                         mean - std,
                         mean + std,
                         color="blue",
                         alpha=0.3,
                         label="± std")

    axes[i].set_title(titles[i])
    axes[i].set_ylabel(labels[i])
    if i == 0:
        axes[i].legend()

axes[2].set_xlabel("timestep")

plt.tight_layout()
plt.savefig(os.path.join(load_path, "all_ee_xyz_with_mean_std.png"))
plt.close()

print("Saved all_ee_xyz_with_mean_std.png (trajectories + mean ± std)")
