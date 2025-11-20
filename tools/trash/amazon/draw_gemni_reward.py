import cv2
import os
import re
import numpy as np
import matplotlib.pyplot as plt

# Paths
scores_dir = "logs/scores"  # folder containing frame_xxx_score_yy.jpg
out_video = scores_dir + "/bottle3_video.mp4"

# Collect frames and parse scores
frames = []
scores = []
for fname in sorted(os.listdir(scores_dir)):
    match = re.match(r"frame_(\d+)_score_([0-9.]+)\.jpg", fname)
    if match:
        frame_id = int(match.group(1))
        score = float(match.group(2))
        frames.append((frame_id, os.path.join(scores_dir, fname)))
        scores.append(score)

# Video properties from first frame
first_img = cv2.imread(frames[0][1])
h, w, _ = first_img.shape
plot_w = int(w * 0.8)  # make plot narrower than frame
out_w = w + plot_w
out_h = h

# Video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = 20
out = cv2.VideoWriter(out_video, fourcc, fps, (out_w, out_h))

# Create video
for i, (fid, fpath) in enumerate(frames):
    img = cv2.imread(fpath)

    # Plot scores up to current frame
    plt.figure(figsize=(4, 4))
    plt.plot(scores[:i + 1], color="blue")
    plt.ylim(0, max(scores) + 1)
    plt.xlim(0, len(frames))
    plt.xlabel("Frame index")
    plt.ylabel("Score")
    plt.title("Task Completion Score")
    plt.tight_layout()
    plt.savefig("logs/tmp_plot.png")
    plt.close()

    # Load plot and resize to match height
    plot_img = cv2.imread("logs/tmp_plot.png")
    plot_img = cv2.resize(plot_img, (plot_w, h))

    # Concatenate: left = frame, right = plot
    combined = np.hstack((img, plot_img))

    out.write(combined)

out.release()
print(f"Saved video: {out_video}")
