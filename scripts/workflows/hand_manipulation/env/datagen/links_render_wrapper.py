from tools.draw.mesh_visualizer.hand_mesh_synthesize import SynthesizeRealRobotPC
import os
import torch

import pickle

import cv2

import numpy as np


class RenderWrapper:

    def __init__(self, env, env_cfg, args_cli):
        self.env = env
        self.args_cli = args_cli
        self.load_path = args_cli.load_path
        load_list = os.listdir(self.load_path)
        self.save_path = args_cli.save_path
        self.num_demos = args_cli.num_demos
        self.load_list = [
            f for f in load_list
            if os.path.isdir(os.path.join(self.load_path, f))
        ]
        os.makedirs(self.save_path, exist_ok=True)
        target_link_name = [
            "palm_lower", "mcp_joint", "pip", "dip", "fingertip",
            "mcp_joint_2", "dip_2", "fingertip_2", "mcp_joint_3", "pip_3",
            "dip_3", "fingertip_3", "thumb_temp_base", "thumb_pip",
            "thumb_dip", "thumb_fingertip", "pip_2", "thumb_right_temp_base"
        ]
        mesh_dir = "source/assets/robot/leap_hand_v2/glb_mesh/"

        self.synthesize_pc = SynthesizeRealRobotPC(mesh_dir, target_link_name)

    def reset(self):
        for i in range(20):
            self.env.step(
                torch.as_tensor(self.env.action_space.sample() * 0.0).to(
                    self.env.device))

    def step_env(self):

        for load_name in self.load_list:
            num_demos = min([
                self.num_demos,
                len(os.listdir(os.path.join(self.load_path, load_name)))
            ])
            for demo_id in range(num_demos):
                demo_path = os.path.join(self.load_path, load_name,
                                         f"episode_{demo_id}.npz")
                save_path = os.path.join(self.save_path, load_name,
                                         f"episode_{demo_id:04d}")
                os.makedirs(save_path, exist_ok=True)
                self.reset()

                try:
                    data = torch.load(demo_path, pickle_module=pickle)
                except:
                    print(f"Failed to load {demo_path}")
                    continue
                actions = torch.cat(data['actions'])
                link_pose = []

                for step_id in range(len(actions)):
                    action = torch.zeros((1, 22)).to(self.env.device)
                    action[..., -16:] = torch.as_tensor(
                        actions[step_id, -16:]).to(self.env.device)
                    self.env.step(action)
                    body_state = self.env.scene[
                        "right_hand"]._data.body_state_w[..., :7]
                    link_pose.append(body_state.cpu().numpy())
                    link_names = self.env.scene["right_hand"].body_names

                link_pose = np.concatenate(link_pose, axis=0)

                link_sorted_pose = []
                target_link_name = []

                for _, link_name in enumerate(
                        list(self.synthesize_pc.mesh_dict.keys())):

                    if link_name not in link_names:
                        continue
                    target_link_name.append(link_name)

                    index = link_names.index(link_name)

                    trajectories_pose = link_pose[:, index]
                    link_sorted_pose.append(trajectories_pose)

                link_sorted_pose = np.array(link_sorted_pose).transpose(
                    1, 0, 2)
                np.save(os.path.join(save_path, "link_sorted_pose.npy"),
                        link_sorted_pose)

                # for i in range(len(link_sorted_pose)):

                #     color = self.synthesize_pc.render_pose(link_sorted_pose[i],
                #                                            target_link_name,
                #                                            render_object=False)
                #     cv2.imwrite(os.path.join(save_path, f"{i:05d}.png"),
                #                 color[..., :3][..., ::-1])

    def close(self):
        self.env.close()
