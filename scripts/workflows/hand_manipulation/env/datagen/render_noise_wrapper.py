from tools.draw.mesh_visualizer.hand_mesh_synthesize import SynthesizeRealRobotPC
import os
import torch

import pickle

import cv2

import numpy as np
import sys

sys.path.append("submodule/diffusion_policy")
import dill
from diffusion_policy.workspace.base_workspace import BaseWorkspace
import hydra
import yaml


class RenderNoiseWrapper:

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
        self.device = env.device

        self.synthesize_pc = SynthesizeRealRobotPC(mesh_dir, target_link_name)
        self.load_diffusion_model()

    def load_diffusion_model(self):
        checkpoint = os.path.join(self.args_cli.diffusion_path, "checkpoints",
                                  "latest.ckpt")

        payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)

        cfg = payload['cfg']

        cfg._target_ = "scripts.workflows.hand_manipulation.utils.diffusion.train_cfm_unet_hand_policy.TrainCFMUnetLowdimWorkspace"
        cfg.policy.num_inference_steps = 3
        cls = hydra.utils.get_class(cfg._target_)

        workspace = cls(cfg, )
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        # get policy from workspace
        self.diffusion_model = workspace.model
        if cfg.training.use_ema:
            self.diffusion_model = workspace.ema_model

        device = torch.device(self.device)
        self.diffusion_model.to(device)
        self.diffusion_model.eval()
        self.chunk_size = self.diffusion_model.n_obs_steps
        self.joint_limits = torch.as_tensor(
            [[-0.314, 2.23], [-0.349, 2.094], [-0.314, 2.23], [-0.314, 2.23],
             [-1.047, 1.047], [-0.46999997, 2.4429998], [-1.047, 1.047],
             [-1.047, 1.047], [-0.5059999, 1.8849999], [-1.2, 1.8999999],
             [-0.5059999, 1.8849999], [-0.5059999, 1.8849999],
             [-0.366, 2.0419998], [-1.34, 1.8799999], [-0.366, 2.0419998],
             [-0.366, 2.0419998]],
            dtype=torch.float32).to(self.device)

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
                teleop_actions = torch.cat(data['actions'])[::10]

                hand_arm_action = torch.zeros(
                    (self.env.num_envs, 22)).to(self.env.device)
                hand_arm_action[:len(teleop_actions),
                                -16:] = teleop_actions[:min(
                                    [len(teleop_actions), self.env.num_envs]),
                                                       -16:].to(
                                                           self.env.device)
                link_pose = []

                self.env.step(hand_arm_action)
                body_state = self.env.scene["right_hand"]._data.body_state_w[
                    ..., :7]
                link_pose.append(body_state.cpu().numpy()[None])

                joint_min = self.joint_limits[:, 0]  # shape [16]
                joint_max = self.joint_limits[:, 1]  # shape [16]
                normalized_joints = (
                    (hand_arm_action[:, -16:] - joint_min[None, :]) /
                    (joint_max - joint_min)[None, :]) * 2 - 1

                obs_dict = {"obs": normalized_joints.unsqueeze(1)}

                for i in range(5):
                    predict_action = self.diffusion_model.predict_action(
                        obs_dict)["action_pred"]
                    for j in range(2):

                        hand_arm_action[:, -16:] = (
                            predict_action[:, j] +
                            1) / 2 * (joint_max - joint_min) + joint_min
                        self.env.step(hand_arm_action)

                        body_state = self.env.scene[
                            "right_hand"]._data.body_state_w[..., :7]
                        link_pose.append(body_state.cpu().numpy()[None])

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

                    trajectories_pose = link_pose[:, :, index]
                    link_sorted_pose.append(trajectories_pose)

                link_sorted_pose = np.array(link_sorted_pose).transpose(
                    1, 0, 2, 3)
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
