import os
import h5py
import torch
from dgl.geometry import farthest_point_sampler
import shutil

# Path to the original file
original_file_path = "logs/1103_grasp2/grasp_normalized.hdf5"
# Path to the copied file
copied_file_path = "logs/1103_grasp2/grasp_normalized_fps.hdf5"

# Copy the original file to create a modified version
shutil.copyfile(original_file_path, copied_file_path)

# Open the copied file
with h5py.File(copied_file_path, 'r+') as modified_grasp:
    for i in range(1):
        print(f"Processing demo_{i}")
        import pdb
        pdb.set_trace()

        # Access the point cloud data for the current demo
        data = modified_grasp["data"][f"demo_{i}"]
        pcd = torch.as_tensor(data["obs"]["seg_pc"]).to("cuda")

        # # Perform farthest point sampling
        # for points in pcd:
        import time
        start = time.time()
        print('====================================')

        # Sample 6000 points using FPS

        index = farthest_point_sampler(pcd[..., :3], 6000)
        indices_expanded = index.unsqueeze(-1).expand(-1, -1, pcd.size(2))
        output = torch.gather(pcd, 1, indices_expanded)

        # Measure and print time taken
        end = time.time()
        print(f"FPS took {end - start:.2f} seconds")
        data["obs"].pop("seg_pc", None)

        # Replace the original point cloud data with the sampled points
        data["obs"]["seg_pc"] = output.cpu().numpy()
        print("Updated point cloud data in the modified file.")

print("Process completed. Saved the modified file.")
