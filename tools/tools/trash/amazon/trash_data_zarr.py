import zarr
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import imageio
import cv2

load_path = "logs/data_0705/retarget_visionpro_data/rl_data/image/bunny"
zarr_files = sorted(
    [f for f in os.listdir(load_path) if f.endswith(".zarr")],
    key=lambda x: int(re.search(r"\d+", x).group())  # extract number
)
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
