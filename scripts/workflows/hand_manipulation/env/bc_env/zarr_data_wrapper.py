import torch

from scripts.workflows.utils.robomimc_collector import RobomimicDataCollector, sample_train_test, fps_points
import json
import h5py
import os
import shutil
import numpy as np

# from dgl.geometry import farthest_point_sampler
import isaaclab.utils.math as math_utils
import copy
import imageio
import gzip
import zarr
from torchvision import transforms
from tools.visualization_utils import vis_pc, visualize_pcd


class ZarrDatawrapper:

    def __init__(self,
                 args_cli,
                 env_config,
                 filter_keys=[],
                 zarr_cfg=None,
                 num_pcd=1):

        self.args_cli = args_cli

        self.env_config = env_config
        self.filter_keys = filter_keys
        self.load_path = args_cli.log_dir
        self.traj_count = 0

        if zarr_cfg is not None:
            self.zarr_cfg = zarr_cfg
            self.obs_key = self.zarr_cfg.dataset.obs_key
            try:
                self.image_key = self.zarr_cfg.dataset.image_key
            except:
                self.image_key = ["image"]
            self.num_pcd = num_pcd
        else:
            self.obs_key = []
            self.image_key = []
            self.zarr_cfg = None

    def load_data(self):
        load_data_success = False
        while not load_data_success:
            try:
                zarr_path = os.path.join(
                    self.args_cli.log_dir,
                    #  self.args_cli.target_object_name,
                    f"episode_{self.traj_count}.zarr")
                load_data_success = True
            except:
                self.traj_count += 1

        data = zarr.open(zarr_path, mode='r')
        data_dict = {}

        # Get all keys (recursive, full path)
        keys = []
        data.visititems(lambda k, v: keys.append((k, v)))
        rgb_buffer = False

        for key, value in keys:
            if isinstance(value, zarr.Array):
                base_key = key.split("/")[-1]

                if base_key in self.image_key:
                    if "rgb" in base_key:
                        # Resize and preprocess image arrays
                        self.image_transform = transforms.Resize(
                            self.zarr_cfg.resize_shape)

                        resized_batch = self.image_transform(
                            torch.from_numpy(np.array(value)).permute(
                                0, 3, 1,
                                2)).numpy()  # (N, H, W, C) -> (N, C, H, W)
                        data_dict[base_key] = resized_batch / 255.0

                    elif base_key in ["seg_pc"]:

                        # points_index = torch.randperm(pcd.shape[-2]).to(
                        #     self.device)

                        # sampled_pcd.append(
                        #     pcd[:, points_index[:self.diffusion_env.
                        #                         image_dim[-1]]])

                        data_dict[base_key] = math_utils.fps_points(
                            torch.as_tensor(value[..., :3]),
                            self.num_pcd).numpy().transpose(0, 2, 1)

                        # result = vis_pc(data_dict[base_key][0].transpose(1, 0))
                        # visualize_pcd([result])

                else:
                    data_dict[base_key] = np.array(value)
        self.traj_count += 1

        return data_dict
