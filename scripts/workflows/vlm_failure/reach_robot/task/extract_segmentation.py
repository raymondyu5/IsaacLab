import numpy as np

import argparse
import cv2
import os
import imageio
from scripts.workflows.vlm_failure.reach_robot.task.joint_pyrender import RenderJoint
# add argparse arguments
import argparse

parser = argparse.ArgumentParser(description="Extract the dataset.")

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
    default=[128, 128],
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

# Path to load data
parser.add_argument(
    "--load_path",
    type=str,
    required=True,  # Ensure the argument is mandatory
    help="Path to load dataset from"
)

# Path to save processed data
parser.add_argument(
    "--save_path",
    type=str,
    required=True,
    help="Path to save the extracted segmentation data"
)

# Log directory path (Fixed issue where it was missing)
parser.add_argument(
    "--log_dir",
    type=str,
    required=True,
    help="Path to log directory for storing run-time logs"
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

args_cli = parser.parse_args()




class Extractor:

    def __init__(
        self,
        args_cli,
    ):
        self.args_cli = args_cli
        self.load_path = args_cli.log_dir + "/" + args_cli.load_path
        self.save_path = args_cli.log_dir + "/" + args_cli.save_path
        os.makedirs(self.save_path, exist_ok=True)
        
        
        self.joint_renderer = RenderJoint(
            args_cli,
            args_cli.save_path,
            args_cli.load_path,
            args_cli.save_video,
            args_cli.panda_link_names,
        )
       

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

        # Initialize an empty boolean mask
        seg_masks = np.zeros_like(rgb, dtype=bool)  # Assuming RGB image

        for id, seg_name in id2lables[0].items():

            if len(seg_name.split("/")) < 6:
                continue
            if seg_name.split("/")[5] in self.args_cli.seg_type:
                seg_mask = seg == id  # Create a boolean mask where the condition is met
                seg_masks |= seg_mask  # Combine masks using logical OR

        # Convert to uint8 if needed for visualization (255 for mask, 0 otherwise)
        # Ensure seg_masks is a boolean mask
        seg_masks = seg_masks > 0  # Convert to boolean if needed
        seg_rgb = np.zeros_like(rgb)

        seg_rgb[seg_masks] = rgb[seg_masks]

        return seg_rgb

    def convert(self):
        for file in os.listdir(f"{self.load_path}"):
            if file.endswith(".npz"):
                if self.args_cli.save_img:
                    os.makedirs(f"{self.args_cli.log_dir}/video",
                                exist_ok=True)
                    writer = imageio.get_writer(
                        f"{self.args_cli.log_dir}/video/{file}.mp4", fps=30)
                data = np.load(f"{self.load_path}/{file}", allow_pickle=True)
                print(f"{self.load_path}/{file}")
                new_obs_buffer = []
               
                obs_buffer = data["obs"]
                for obs in obs_buffer:
                    rgb = obs["rgb"]
                    seg = obs["segmentation"]
                    id2lables = obs["id2lables"]
                    

                    seg_rgb = self.seg_on_rgb(rgb, seg, id2lables)
                    seg_rgb_reshaped = np.concatenate(seg_rgb, axis=1)
                    rgb_reshaped = np.concatenate(rgb, axis=1)
                  
                    render_joint_image = self.joint_renderer.extract(obs)
                    obs["sphere_image"] = render_joint_image
                    
                 
                    
                    if self.args_cli.save_img:
                        render_joint_image_reshaped = np.concatenate(render_joint_image, axis=1)
                        
            
                        mask = np.all(render_joint_image_reshaped == [255, 255, 255], axis=-1)

                        # Replace white pixels in rendered_img with corresponding pixels from rgb_image
                        mask_combined_img = np.where(mask[..., None], seg_rgb_reshaped,
                                                render_joint_image_reshaped)
                       
                        combined_image = np.concatenate([seg_rgb_reshaped, mask_combined_img], axis=0)
                        writer.append_data(combined_image)
                    new_obs_buffer.append(obs)  
               
                data = {"obs": new_obs_buffer, "actions": data["actions"]}
                
                np.savez_compressed(f"{self.save_path}/{file}", **data)


extractor = Extractor(args_cli)

extractor.convert()
