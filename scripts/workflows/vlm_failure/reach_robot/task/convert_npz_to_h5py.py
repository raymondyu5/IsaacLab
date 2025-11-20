from tools.visualization_utils import *

from isaaclab.app import AppLauncher
import cv2
import matplotlib.pyplot as plt
import numpy as np
# import open3d as o3d
from scripts.workflows.utils.parse_setting import save_params_to_yaml, parser
import pickle
import json
from scripts.workflows.vlm_failure.reach_robot.task.joint_pyrender import RenderJoint
parser.add_argument("--use_failure",
                    action="store_true",
                    default=False,
                    help="Disable fabric and use USD I/O operations.")


# Boolean flag to save video
parser.add_argument(
    "--save_video",
    action="store_true",
    default=False,
    help="Enable video saving mode"
)

# Floating point camera scale
parser.add_argument(
    "--camera_scale",
    type=float,
    default=0.06,
    help="Scale factor for camera view"
)

# List of integers for resolution (Fixed: Using nargs instead of type=list)
parser.add_argument(
    "--resolution",  # Fixed typo from 'resoultion' to 'resolution'
    nargs=2,
    type=int,
    default=[256, 256],
    help="Resolution of the camera output as two integers (width height)"
)

# List of strings for specific Panda robot links
parser.add_argument(
    "--panda_link_names",
    nargs="+",  # Accept multiple values from CLI
    type=str,
    default=[
        "panda_link1",
        "panda_link4",
        "panda_link6",
        "panda_hand",
        "mug",
        "yellow_cube"
    ],
    help="List of Panda robot links to track"
)




parser.add_argument(
    "--save_img",
    action="store_true",    
    default=False,
    help="Path to log directory for storing run-time logs"
)
# List of segmentation types (Fixed: Using nargs instead of type=list)
parser.add_argument(
    "--seg_type",
    nargs="+",
    type=str,
    default=["all"],
    help="Segmentation types to extract (e.g., panda_hand, panda_leftfinger, panda_rightfinger)"
)

AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# # # launch omniverse app

import json
import h5py
import os
import shutil
import numpy as np
from scripts.workflows.utils.parse_setting import parser
import imageio
args_cli = parser.parse_args()
import torch


def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert NumPy arrays to lists
    if isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)  # Convert NumPy floats to Python floats
    if isinstance(obj, np.int32) or isinstance(obj, np.int64):
        return int(obj)  # Convert NumPy ints to Python ints
    return obj


class Converter:

    def __init__(
        self,
        args_cli,
        filter_keys,
        noramilize_action=False,
    ):
        self.args_cli = args_cli
        self.load_path = args_cli.log_dir + "/" + args_cli.load_path
        self.save_path = args_cli.log_dir + "/" + args_cli.save_path
        os.makedirs(self.save_path, exist_ok=True)

        self.end_name = args_cli.end_name
        self.add_noise = args_cli.noise_pc
        self.filter_keys = filter_keys

        
        self.noramilize_action = noramilize_action
        
        self.joint_renderer = RenderJoint(
            args_cli,
            args_cli.save_path,
            args_cli.load_path,
            args_cli.save_video,
            args_cli.panda_link_names,
        )

    def filter_obs_buffer(self, obs_buffer, index, obs_name="obs"):
        if "policy" in obs_buffer[index]:
            obs = obs_buffer[index]["policy"]
        else:
            obs = obs_buffer[index]
        new_obs = {}

        for key, value in obs.items():
            if key in self.filter_keys:
                continue
            if isinstance(value, torch.Tensor):

                new_obs[key] = value.cpu().numpy()[0]
            else:

                new_obs[key] = value
        return new_obs
    
    def seg_on_rgb(self, rgb, seg, id2lables):

        if "all" in self.args_cli.seg_type:
            seg_masks = np.zeros_like(rgb, dtype=bool)  # Assuming RGB image

            for id, seg_name in id2lables[0].items():

                if len(seg_name.split("/")) < 6:
                    continue
                if "panda" in seg_name.split("/")[5]:
                    seg_mask = seg == id  # Create a boolean mask where the condition is met
                    seg_masks |= seg_mask  # Combine masks using logical OR
                elif seg_name.split("/")[4]  in self.args_cli.panda_link_names:
                    seg_mask = seg == id  # Create a boolean mask where the condition is met
                    seg_masks |= seg_mask  # Combine masks using logical OR

                # Convert to uint8 if needed for visualization (255 for mask, 0 otherwise)
                # Ensure seg_masks is a boolean mask
                seg_masks = seg_masks > 0  # Convert to boolean if needed
                seg_rgb = np.zeros_like(rgb)

                seg_rgb[seg_masks] = rgb[seg_masks]

            return seg_rgb


    def convert(self):

        # self.env_config = np.load(f"{self.load_path}/env_setting.npy",
        #                           allow_pickle=True)

        # config = self.env_config.item()

        # # Use the `default` parameter of json.dumps to handle non-serializable objects
        # json_string = json.dumps(config,
        #                          default=convert_to_serializable,
        #                          indent=4)
        num_demos = 0
        for file in os.listdir(f"{self.load_path}{self.end_name}"):
            if self.args_cli.num_demos == num_demos:
                break
            if file.endswith(".npz"):
                data = torch.load(os.path.join(
                    f"{self.load_path}{self.end_name}", file),
                                  pickle_module=pickle)
                if self.args_cli.save_img:
                    os.makedirs(f"{self.args_cli.log_dir}/video",
                                exist_ok=True)
                    writer = imageio.get_writer(
                        f"{self.args_cli.log_dir}/video/{file}.mp4", fps=30)
                num_demos += 1

                obs_buffer = data["obs"]
                actions_buffer = torch.cat(data["actions"],
                                           dim=0).cpu().numpy()
                new_obs_buffer = []

                for index in range(len(obs_buffer)):
                   
                    render_joint_image = self.joint_renderer.extract(obs_buffer[index]["policy"])
                    
                    rgb = obs_buffer[index]["policy"]["rgb"][0].cpu().numpy()
                    segmentation = obs_buffer[index]["policy"]["segmentation"][0].cpu().numpy()
                    
                    seg_rgb = self.seg_on_rgb(rgb, segmentation, obs_buffer[index]["policy"]["id2lables"])
                    seg_rgb_reshaped = np.concatenate(seg_rgb, axis=1)
                    rgb_reshaped = np.concatenate(rgb, axis=1)
                  
               
                    if self.args_cli.save_img:
                        render_joint_image_reshaped = np.concatenate(render_joint_image, axis=1)
                        
            
                        mask = np.all(render_joint_image_reshaped == [255, 255, 255], axis=-1)
                       

                        # Replace white pixels in rendered_img with corresponding pixels from rgb_image
                        mask_combined_img = np.where(mask[..., None], seg_rgb_reshaped,render_joint_image_reshaped)
                       
                        combined_image = np.concatenate([seg_rgb_reshaped, mask_combined_img], axis=0)
                        writer.append_data(combined_image)
                    
                  
                    new_obs = self.filter_obs_buffer(obs_buffer,   index, obs_name="obs")
                    new_obs["sphere_image"] = render_joint_image
                    new_obs_buffer.append(new_obs)
               
                data = {"obs": new_obs_buffer, "actions": actions_buffer}
               
                np.savez(f"{self.save_path}/{file}", **data)
                # data = np.load(f"{self.save_path}/{file}", allow_pickle=True)


Converter(args_cli, filter_keys=["seg_rgb"], noramilize_action=True).convert()
