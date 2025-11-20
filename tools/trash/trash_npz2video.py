import torch
import os

source_dir = "/home/ensu/Documents/weird/IsaacLab/logs/kitchen02_yunchu"
import pickle
import imageio

os.makedirs(f"{source_dir}/videos", exist_ok=True)
for file in os.listdir(f"{source_dir}/render_data"):

    if file.endswith(".npz"):
        name = file.split(".")[0]
        video_writer = imageio.get_writer(f"{source_dir}/videos/{name}.mp4",
                                          fps=30)

        data = torch.load(os.path.join(f"{source_dir}/render_data", file),
                          pickle_module=pickle)

        obs_buffer = data["obs"]

        for obs in obs_buffer:

            images = obs["rgb"][0]
            con_image = torch.cat([images[i] for i in range(len(images))],
                                  dim=1).cpu().numpy()
            video_writer.append_data(con_image)
