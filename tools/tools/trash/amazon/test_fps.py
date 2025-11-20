import fpsample
import numpy as np
import time
# Generate random point cloud
pc = np.random.rand(40960, 3)
## sample 1024 points
fps_samples_idx = fpsample.fps_sampling(pc, 1024)
start_time = time.time()
# Vanilla FPS
fps_samples_idx = fpsample.fps_sampling(pc, 1024)
print(f"FPS sampling time: {time.time() - start_time:.4f} seconds")
# FPS + NPDU
start_time = time.time()
fps_npdu_samples_idx = fpsample.fps_npdu_sampling(pc, 1024)
print(f"FPS sampling time: {time.time() - start_time:.4f} seconds")
# FPS + NPDU + KDTree
start_time = time.time()
fps_npdu_kdtree_samples_idx = fpsample.fps_npdu_kdtree_sampling(pc, 1024)
print(f"FPS sampling time: {time.time() - start_time:.4f} seconds")
# KDTree-based FPS
start_time = time.time()
kdtree_fps_samples_idx = fpsample.bucket_fps_kdtree_sampling(pc, 1024)
print(f"FPS sampling time: {time.time() - start_time:.4f} seconds")
# NOTE: Probably the best
# Bucket-based FPS or QuickFPS
start_time = time.time()
kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(pc, 1024, h=3)
print(f"FPS sampling time: {time.time() - start_time:.4f} seconds")
import isaaclab.utils.math as math_utils
import torch

torch_data = torch.as_tensor(pc[None]).to("cuda")

start_time = time.time()
math_utils.fps_points(torch_data, 1024)
torch.cuda.synchronize()
print(f"FPS points time: {time.time() - start_time:.4f} seconds")
from torch_geometric.nn import fps

start_time = time.time()
index = fps(torch_data,
            torch.zeros(len(torch_data)).to(torch_data.device).to(torch.int64),
            ratio=0.025)
torch.cuda.synchronize()
print(f"FPS points time: {time.time() - start_time:.4f} seconds")
