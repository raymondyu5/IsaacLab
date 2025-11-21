# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Script to visualize zarr data collected by collect_rl_data.py."""

import argparse
import os
import zarr
import numpy as np
import imageio
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless
import matplotlib.pyplot as plt


def load_rgb_data(zarr_path):
    """Load RGB images from a zarr file.
    
    Args:
        zarr_path: Path to the zarr file/directory
        
    Returns:
        RGB images as numpy array
    """
    if not os.path.exists(zarr_path):
        raise FileNotFoundError(f"Zarr file not found: {zarr_path}")
    
    data = zarr.open(zarr_path, mode='r')
    
    # Load RGB images
    rgb_channels = []
    for i in range(10):  # Check up to 10 channels
        key = f'data/rgb_{i}'
        if key in data:
            rgb_channels.append(np.array(data[key]))
    
    if not rgb_channels:
        raise ValueError("No RGB images found in zarr file")
    
    # Stack RGB channels: (T, H, W) for each channel -> (T, H, W, 3)
    rgb = np.stack(rgb_channels[:3], axis=-1)
    
    # Handle unusual shapes like (T, H, W, 3, 1) -> (T, H, W, 3)
    while len(rgb.shape) > 4:
        rgb = rgb.squeeze(axis=-1)
    
    print(f"Loaded RGB images: shape {rgb.shape}")
    
    return rgb


def save_rgb_video(zarr_path, output_path=None, fps=10):
    """Save RGB images from zarr data as mp4 video.
    
    Args:
        zarr_path: Path to the zarr file
        output_path: Path to save the output video (default: same as zarr_path with .mp4 extension)
        fps: Frames per second for the video
    """
    # Load RGB data
    rgb_images = load_rgb_data(zarr_path)
    
    num_timesteps = rgb_images.shape[0]
    print(f"Saving {num_timesteps} frames to video")
    
    # Set output path
    if output_path is None:
        base_name = os.path.splitext(zarr_path)[0]
        if base_name.endswith('.zarr'):
            base_name = base_name[:-5]
        output_path = f"{base_name}_video.mp4"
    
    # Create video writer
    video_writer = imageio.get_writer(output_path, fps=fps)
    
    # Write each frame
    for t in range(num_timesteps):
        frame = rgb_images[t]
        
        # Ensure frame is uint8 and in correct format
        if frame.dtype != np.uint8:
            # Normalize if needed
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        
        # Ensure shape is (H, W, 3)
        if len(frame.shape) == 2:
            # Grayscale, convert to RGB
            frame = np.stack([frame] * 3, axis=-1)
        elif len(frame.shape) == 3 and frame.shape[2] == 1:
            # Single channel, convert to RGB
            frame = np.repeat(frame, 3, axis=2)
        
        video_writer.append_data(frame)
        
        if (t + 1) % 10 == 0:
            print(f"Processed {t + 1}/{num_timesteps} frames")
    
    # Close video writer
    video_writer.close()
    print(f"Saved video to: {output_path}")


def visualize_pointcloud(zarr_path, timestep=0, max_points=5000, output_path=None):
    """Visualize point cloud from zarr data using matplotlib.
    
    Args:
        zarr_path: Path to the zarr file
        timestep: Which timestep to visualize (default: 0)
        max_points: Maximum number of points to display (default: 5000)
        output_path: Path to save the image (if None, displays interactively)
    """
    if not os.path.exists(zarr_path):
        raise FileNotFoundError(f"Zarr file not found: {zarr_path}")
    
    data = zarr.open(zarr_path, mode='r')
    
    # Load point cloud
    if 'seg_pc' not in data['data']:
        raise ValueError("No point cloud data (seg_pc) found in zarr file")
    
    point_clouds = np.array(data['data/seg_pc'])  # Shape: (T, N, 3) or (T, C, N, 3)
    
    # Handle different point cloud shapes
    if len(point_clouds.shape) == 4:
        # Shape: (T, C, N, 3) - take first camera/view
        point_clouds = point_clouds[:, 0, :, :]
    elif len(point_clouds.shape) == 3:
        # Shape: (T, N, 3) - already correct
        pass
    else:
        raise ValueError(f"Unexpected point cloud shape: {point_clouds.shape}")
    
    num_timesteps = point_clouds.shape[0]
    if timestep >= num_timesteps:
        raise ValueError(f"Timestep {timestep} out of range (max: {num_timesteps - 1})")
    
    # Get point cloud for this timestep
    pc = point_clouds[timestep]  # Shape: (N, 3)
    
    # Remove any NaN or invalid points
    valid_mask = ~np.isnan(pc).any(axis=1)
    if len(pc.shape) > 1:
        valid_mask = valid_mask & (np.abs(pc).max(axis=1) < 1e6)  # Remove extreme outliers
    pc = pc[valid_mask]
    
    # Subsample if too many points
    if len(pc) > max_points:
        indices = np.random.choice(len(pc), max_points, replace=False)
        pc = pc[indices]
        print(f"Subsampled from {len(valid_mask)} to {max_points} points")
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1, alpha=0.6, c=pc[:, 2], cmap='viridis')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Point Cloud - Timestep {timestep} ({len(pc)} points)')
    
    # Set equal aspect ratio
    max_range = np.array([pc[:, 0].max() - pc[:, 0].min(),
                          pc[:, 1].max() - pc[:, 1].min(),
                          pc[:, 2].max() - pc[:, 2].min()]).max() / 2.0
    mid_x = (pc[:, 0].max() + pc[:, 0].min()) * 0.5
    mid_y = (pc[:, 1].max() + pc[:, 1].min()) * 0.5
    mid_z = (pc[:, 2].max() + pc[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved point cloud visualization to: {output_path}")
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Save RGB video from zarr data collected by collect_rl_data.py")
    parser.add_argument(
        "--zarr_path",
        type=str,
        required=True,
        help="Path to the zarr file (e.g., logs/rl_data_collection/ycb/pcd_residual_chunk/tomato_soup_can/episode_62.zarr)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the video (default: same as zarr_path with _video.mp4 extension)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for the output video (default: 10)"
    )
    parser.add_argument(
        "--visualize_pc",
        action="store_true",
        help="Visualize point cloud instead of saving video"
    )
    parser.add_argument(
        "--timestep",
        type=int,
        default=0,
        help="Timestep to visualize (for point cloud visualization, default: 0)"
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=5000,
        help="Maximum number of points to display (default: 5000)"
    )
    parser.add_argument(
        "--pc_output",
        type=str,
        default=None,
        help="Output path for point cloud image (default: same as zarr_path with _pc_t{timestep}.png)"
    )
    
    args = parser.parse_args()
    
    if args.visualize_pc:
        if args.pc_output is None:
            base_name = 'vis'
            args.pc_output = f"{base_name}/pc_t{args.timestep}.png"
        
        visualize_pointcloud(
            args.zarr_path,
            timestep=args.timestep,
            max_points=args.max_points,
            output_path=args.pc_output
        )
    else:
        save_rgb_video(
            args.zarr_path,
            output_path=args.output,
            fps=args.fps
        )


if __name__ == "__main__":
    main()

