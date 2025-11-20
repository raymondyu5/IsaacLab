#!/usr/bin/env python3
"""Headless visualization script for zarr episodes - exports to MP4 or PNG frames.

Usage:
  python visualize_episode_headless.py --episode /path/to/episode_2.zarr --output video.mp4
  python visualize_episode_headless.py --episode /path/to/episode_2.zarr --output-dir frames/

Features:
- Non-interactive, headless rendering (works in apptainer/container)
- Exports to MP4 (requires imageio-ffmpeg) or PNG frames (no dependencies)
- 4-panel layout: RGB, rewards, 3D point cloud, XZ side view
- Configurable frame range and FPS

Dependencies: zarr, numpy, matplotlib, imageio (optional, for MP4)
"""
import argparse
import os
import sys
import numpy as np
import zarr
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm


def safe_get(group, name):
    if name in group:
        return group[name]
    return None


def to_array(z):
    if z is None:
        return None
    try:
        return z[:]  # load into memory
    except Exception:
        return np.array(z)


class HeadlessEpisodeRenderer:
    def __init__(self, episode_path, max_points=20000):
        # Try to open zarr file - handle both v2 and v3 formats
        try:
            # First try direct open (works for zarr v2 format)
            g = zarr.open_group(episode_path, mode="r")
        except Exception as e:
            # If that fails, the file might be zarr v3 format
            # Convert from v3 to v2 by using DirectoryStore explicitly
            import zarr.storage
            print(f"Note: Standard zarr.open_group failed ({e}), trying DirectoryStore...")
            store = zarr.storage.DirectoryStore(episode_path)
            g = zarr.group(store=store)
        if "data" in g:
            g = g["data"]
        self.g = g

        self.rgb = safe_get(g, "rgb_0")
        self.rewards = safe_get(g, "rewards")
        self.seg = safe_get(g, "seg_pc")

        self.rgb_arr = to_array(self.rgb)
        self.rew_arr = to_array(self.rewards)
        self.seg_arr = to_array(self.seg)

        # infer length
        lengths = []
        for a in (self.rgb_arr, self.rew_arr, self.seg_arr):
            if a is None:
                continue
            if hasattr(a, "shape"):
                lengths.append(a.shape[0])
        self.length = int(max(lengths)) if lengths else 0

        self.max_points = max_points

        # Create figure once
        self.fig = plt.figure(figsize=(14, 5))
        self.ax_img = self.fig.add_subplot(1, 4, 1)
        self.ax_plot = self.fig.add_subplot(1, 4, 2)
        self.ax_3d = self.fig.add_subplot(1, 4, 3, projection="3d")
        self.ax_side = self.fig.add_subplot(1, 4, 4)

    def render_frame(self, idx):
        """Render a single frame."""
        self.ax_img.cla()
        self.ax_plot.cla()
        self.ax_3d.cla()
        self.ax_side.cla()

        # draw image
        if self.rgb_arr is not None and idx < self.rgb_arr.shape[0]:
            img = self.rgb_arr[idx]
            # handle channels last or first
            if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
                img = np.transpose(img, (1, 2, 0))
            self.ax_img.imshow(img.astype("uint8"))
        else:
            self.ax_img.text(0.5, 0.5, "no rgb_0", ha="center", va="center")
        self.ax_img.set_title(f"frame {idx}")
        self.ax_img.axis("off")

        # draw rewards
        if self.rew_arr is not None:
            try:
                r = np.asarray(self.rew_arr).squeeze()
                self.ax_plot.plot(r, "-o", markersize=2)
                self.ax_plot.axvline(idx, color="r")
                self.ax_plot.set_title("rewards over time")
                self.ax_plot.set_xlabel("frame")
                self.ax_plot.set_ylabel("reward")
            except Exception:
                self.ax_plot.text(0.5, 0.5, "unable to plot rewards", ha="center")
        else:
            self.ax_plot.text(0.5, 0.5, "no rewards", ha="center")

        # draw seg point cloud
        if self.seg_arr is not None and idx < self.seg_arr.shape[0]:
            pts = self.seg_arr[idx]
            pts = np.asarray(pts)
            if pts.ndim == 2 and pts.shape[1] >= 3:
                # downsample if too many
                n = pts.shape[0]
                if n > self.max_points:
                    idxs = np.random.choice(n, self.max_points, replace=False)
                    pts = pts[idxs]
                self.ax_3d.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1)
                self.ax_3d.set_title("seg_pc (3D)")
                self.ax_3d.set_xlabel("X")
                self.ax_3d.set_ylabel("Y")
                self.ax_3d.set_zlabel("Z")
            else:
                self.ax_3d.text2D(0.5, 0.5, "seg_pc shape unexpected", transform=self.ax_3d.transAxes, ha="center")
        else:
            self.ax_3d.text2D(0.5, 0.5, "no seg_pc", transform=self.ax_3d.transAxes, ha="center")

        # draw side-view XZ projection
        if self.seg_arr is not None and idx < self.seg_arr.shape[0]:
            try:
                pts_side = np.asarray(self.seg_arr[idx])
                if pts_side.ndim == 2 and pts_side.shape[1] >= 3:
                    n = pts_side.shape[0]
                    if n > self.max_points:
                        idxs = np.random.choice(n, self.max_points, replace=False)
                        pts_side = pts_side[idxs]
                    # X vs Z (side view)
                    self.ax_side.scatter(pts_side[:, 0], pts_side[:, 2], s=1, c=pts_side[:, 2], cmap='viridis')
                    self.ax_side.set_xlabel('X')
                    self.ax_side.set_ylabel('Z')
                    self.ax_side.set_title('Side View (XZ)')
                    # equal aspect and reasonable limits
                    try:
                        max_range = np.array([pts_side[:, 0].max() - pts_side[:, 0].min(),
                                              pts_side[:, 2].max() - pts_side[:, 2].min()]).max() / 2.0
                        mid_x = (pts_side[:, 0].max() + pts_side[:, 0].min()) * 0.5
                        mid_z = (pts_side[:, 2].max() + pts_side[:, 2].min()) * 0.5
                        self.ax_side.set_xlim(mid_x - max_range, mid_x + max_range)
                        self.ax_side.set_ylim(mid_z - max_range, mid_z + max_range)
                        self.ax_side.set_aspect('equal', adjustable='box')
                    except Exception:
                        pass
            except Exception:
                pass

        self.fig.tight_layout()

    def export_png_frames(self, output_dir, start=0, end=None, max_frames=500):
        """Export frames as individual PNG files."""
        end = end if end is not None else self.length
        end = int(min(end, self.length))
        start = int(max(0, start))
        num = end - start

        if max_frames is not None and num > max_frames:
            end = start + max_frames
            num = max_frames

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        print(f"Exporting {num} frames to {output_dir}")

        # Render and save frames
        for i in tqdm(range(num), desc="Rendering frames"):
            self.render_frame(start + i)
            fname = os.path.join(output_dir, f"frame_{start + i:06d}.png")
            self.fig.savefig(fname, dpi=100, bbox_inches='tight')

        print(f'Successfully saved {num} PNG frames to {output_dir}')

    def export_mp4_imageio(self, output_path, start=0, end=None, fps=15, max_frames=500):
        """Export to MP4 using imageio (no external ffmpeg needed)."""
        try:
            import imageio
        except ImportError:
            print("ERROR: imageio not installed. Install with: pip install imageio imageio-ffmpeg")
            print("Falling back to PNG export...")
            output_dir = output_path.replace('.mp4', '_frames')
            self.export_png_frames(output_dir, start, end, max_frames)
            return

        end = end if end is not None else self.length
        end = int(min(end, self.length))
        start = int(max(0, start))
        num = end - start

        if max_frames is not None and num > max_frames:
            end = start + max_frames
            num = max_frames

        print(f"Exporting frames {start}..{end-1} ({num} frames) to {output_path} at {fps} fps")

        try:
            # Use imageio to write video
            writer = imageio.get_writer(output_path, fps=fps, codec='libx264', pixelformat='yuv420p')

            for i in tqdm(range(num), desc="Rendering frames"):
                self.render_frame(start + i)

                # Convert matplotlib figure to numpy array
                self.fig.canvas.draw()
                img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
                img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

                writer.append_data(img)

            writer.close()
            print(f'Successfully saved MP4 to {output_path}')

        except Exception as e:
            print(f'Error during export: {e}')
            print('Falling back to PNG export...')
            output_dir = output_path.replace('.mp4', '_frames')
            self.export_png_frames(output_dir, start, end, max_frames)


def main():
    parser = argparse.ArgumentParser(description="Headless episode visualization - export to MP4 or PNG frames")
    parser.add_argument("--episode", type=str, required=True, help="path to episode zarr")
    parser.add_argument("--output", type=str, default=None, help="output MP4 path (e.g., video.mp4)")
    parser.add_argument("--output-dir", type=str, default=None, help="output directory for PNG frames")
    parser.add_argument("--fps", type=int, default=15, help="frames per second (MP4 only)")
    parser.add_argument("--start", type=int, default=0, help="start frame")
    parser.add_argument("--end", type=int, default=None, help="end frame (exclusive)")
    parser.add_argument("--max-frames", type=int, default=500, help="maximum frames to export")
    parser.add_argument("--max-points", type=int, default=20000, help="max points to render per frame")
    parser.add_argument("--print-shapes", action="store_true", help="print shapes and exit")

    args = parser.parse_args()

    # Create renderer
    renderer = HeadlessEpisodeRenderer(args.episode, max_points=args.max_points)

    if args.print_shapes:
        print("Arrays present:")
        for name in ("rgb_0", "rewards", "seg_pc"):
            a = safe_get(renderer.g, name)
            if a is None:
                print(f"  {name}: MISSING")
            else:
                try:
                    print(f"  {name}: shape={a.shape}, dtype={a.dtype}")
                except Exception:
                    print(f"  {name}: (unable to read shape)")
        print(f"\nTotal frames: {renderer.length}")
        return

    # Determine export mode
    if args.output_dir:
        # Export as PNG frames
        renderer.export_png_frames(args.output_dir, args.start, args.end, args.max_frames)
    elif args.output:
        # Export as MP4 using imageio
        renderer.export_mp4_imageio(args.output, args.start, args.end, args.fps, args.max_frames)
    else:
        print("ERROR: Must specify either --output (MP4) or --output-dir (PNG frames)")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
