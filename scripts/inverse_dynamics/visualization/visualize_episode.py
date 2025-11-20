#!/usr/bin/env python3
"""Visualize a zarr episode directory.

Usage:
  python visualize_episode.py --episode /path/to/episode_2.zarr --frame 0

Features:
- Prints shapes of key arrays when --print-shapes is used
- Interactive viewer: left/right arrows to move frame, space to play/pause

Dependencies: zarr, numpy, matplotlib, tqdm
"""
import argparse
import time
import numpy as np
import zarr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


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


class EpisodeViewer:
    def __init__(self, episode_path, max_points=20000):
        g = zarr.open_group(episode_path, mode="r")
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

        self.idx = 0
        self.playing = False
        self.max_points = max_points

        # setup figure (add extra column for side-view XZ projection)
        self.fig = plt.figure(figsize=(14, 5))
        self.ax_img = self.fig.add_subplot(1, 4, 1)
        self.ax_plot = self.fig.add_subplot(1, 4, 2)
        self.ax_3d = self.fig.add_subplot(1, 4, 3, projection="3d")
        self.ax_side = self.fig.add_subplot(1, 4, 4)

        self.img_obj = None
        self.line_v = None
        self.seg_obj = None
        self.side_obj = None

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.draw()

    def draw(self):
        self.ax_img.cla()
        self.ax_plot.cla()
        self.ax_3d.cla()
        self.ax_side.cla()

        # draw image
        if self.rgb_arr is not None and self.idx < self.rgb_arr.shape[0]:
            img = self.rgb_arr[self.idx]
            # handle channels last or first
            if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
                img = np.transpose(img, (1, 2, 0))
            self.ax_img.imshow(img.astype("uint8"))
        else:
            self.ax_img.text(0.5, 0.5, "no rgb_0", ha="center", va="center")
        self.ax_img.set_title(f"frame {self.idx}")
        self.ax_img.axis("off")

        # draw rewards
        if self.rew_arr is not None:
            try:
                r = np.asarray(self.rew_arr).squeeze()
                self.ax_plot.plot(r, "-o", markersize=2)
                self.line_v = self.ax_plot.axvline(self.idx, color="r")
                self.ax_plot.set_title("rewards over time")
            except Exception:
                self.ax_plot.text(0.5, 0.5, "unable to plot rewards", ha="center")
        else:
            self.ax_plot.text(0.5, 0.5, "no rewards", ha="center")

        # draw seg point cloud
        if self.seg_arr is not None and self.idx < self.seg_arr.shape[0]:
            pts = self.seg_arr[self.idx]
            pts = np.asarray(pts)
            if pts.ndim == 2 and pts.shape[1] >= 3:
                # downsample if too many
                n = pts.shape[0]
                if n > self.max_points:
                    idxs = np.random.choice(n, self.max_points, replace=False)
                    pts = pts[idxs]
                self.ax_3d.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1)
                self.ax_3d.set_title("seg_pc (3D)")
            else:
                self.ax_3d.text(0.5, 0.5, "seg_pc shape unexpected", ha="center")
        else:
            self.ax_3d.text(0.5, 0.5, "no seg_pc", ha="center")
            self.ax_side.text(0.5, 0.5, "no seg_pc", ha="center")

        # draw side-view XZ projection
        if self.seg_arr is not None and self.idx < (self.seg_arr.shape[0] if hasattr(self.seg_arr, 'shape') else self.length):
            try:
                pts_side = np.asarray(self.seg_arr[self.idx])
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
                else:
                    # shape unexpected
                    pass
            except Exception:
                # garbage seg array — ignore side view
                pass

        self.fig.canvas.draw_idle()

    def update_idx(self, new_idx):
        self.idx = int(np.clip(new_idx, 0, max(0, self.length - 1)))
        self.draw()

    def on_key(self, event):
        if event.key in ("right", "d"):
            self.update_idx(self.idx + 1)
        elif event.key in ("left", "a"):
            self.update_idx(self.idx - 1)
        elif event.key == " ":
            self.playing = not self.playing
            if self.playing:
                self._play()
        elif event.key in ("q", "escape"):
            plt.close(self.fig)

    def _play(self):
        while self.playing:
            self.update_idx(self.idx + 1)
            plt.pause(0.03)
            if self.idx >= self.length - 1:
                self.playing = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=str, default="./sim_example_data/episode_2.zarr", help="path to episode zarr")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--max-points", type=int, default=20000)
    parser.add_argument("--print-shapes", action="store_true")
    parser.add_argument("--export-mp4", dest="export_mp4", type=str, default=None,
                        help="path to write an mp4 of the rollout (uses ffmpeg)")
    parser.add_argument("--fps", type=int, default=15, help="frames per second for mp4")
    parser.add_argument("--start", type=int, default=0, help="start frame for export")
    parser.add_argument("--end", type=int, default=None, help="end frame (exclusive) for export")
    parser.add_argument("--max-frames", type=int, default=500, help="maximum frames to export (safety)")
    args = parser.parse_args()

    viewer = EpisodeViewer(args.episode, max_points=args.max_points)

    if args.print_shapes:
        print("arrays present:")
        for name in ("rgb_0", "rewards", "seg_pc"):
            a = safe_get(viewer.g, name)
            if a is None:
                print(f"  {name}: MISSING")
            else:
                try:
                    print(f"  {name}: shape={a.shape}, dtype={a.dtype}")
                except Exception:
                    print(f"  {name}: (unable to read shape)")
        return

    if args.export_mp4:
        out_path = args.export_mp4
        start = args.start
        end = args.end if args.end is not None else viewer.length
        fps = args.fps
        max_frames = args.max_frames
        # clamp
        end = int(min(end, viewer.length))
        start = int(max(0, start))
        num = end - start
        if max_frames is not None and num > max_frames:
            end = start + max_frames
            num = max_frames

        print(f"Exporting frames {start}..{end-1} ({num} frames) to {out_path} at {fps} fps")

        # try to use matplotlib's animation with FFMpegWriter
        try:
            import matplotlib.animation as animation
            # create fig and funcs that update the viewer's index and redraw
            def update_frame(i):
                viewer.update_idx(start + i)

            ani = animation.FuncAnimation(viewer.fig, update_frame, frames=num, blit=False)
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=fps, metadata=dict(artist='visualize_episode'), bitrate=2000000)
            ani.save(out_path, writer=writer)
            print('Saved mp4 via matplotlib+ffmpeg')
        except Exception as e:
            print('Matplotlib+ffmpeg writer failed, falling back to PNG frames + ffmpeg. Error:', e)
            import os, subprocess, tempfile
            tmpd = tempfile.mkdtemp(prefix='viz_frames_')
            print('Writing PNG frames to', tmpd)
            for i in range(num):
                viewer.update_idx(start + i)
                fname = os.path.join(tmpd, f"frame_{i:06d}.png")
                viewer.fig.savefig(fname)
            # run ffmpeg
            ffmpeg_cmd = [
                'ffmpeg', '-y', '-framerate', str(fps), '-i', os.path.join(tmpd, 'frame_%06d.png'),
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', out_path
            ]
            print('Running ffmpeg to combine frames...')
            try:
                subprocess.check_call(ffmpeg_cmd)
                print('Saved mp4 via ffmpeg')
            except Exception as e2:
                print('ffmpeg failed:', e2)
                print('MP4 export failed')

        return

    viewer.update_idx(args.frame)
    plt.show()


if __name__ == "__main__":
    main()
