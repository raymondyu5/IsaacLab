import numpy as np

from scripts.workflows.hand_manipulation.env.bc_env.zarr_replay_env_wrapper import ZarrReplayWrapper
import sys
import os
from torchvision import transforms

sys.path.append("submodule/diffusion_policy")
import dill
from diffusion_policy.workspace.base_workspace import BaseWorkspace
import hydra
from scripts.workflows.hand_manipulation.utils.vae.data_normalizer import (
    TemporalEnsembleBufferAction, TemporalEnsembleBufferObservation,
    TemporalEnsembleImageObservation)
import torch
import copy
from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
import matplotlib.pyplot as plt
from scripts.workflows.hand_manipulation.utils.diffusion.dataset.zarr_buffer import sharpen_batch
from torchvision.transforms import v2 as T


class ImageBCDiffusionWrapper(ZarrReplayWrapper):

    def __init__(self, env, env_cfg, args_cli, replay_env=None):

        self.env = env
        self.args_cli = args_cli
        self.env_cfg = env_cfg
        self.add_left_hand = args_cli.add_left_hand
        self.add_right_hand = args_cli.add_right_hand
        self.device = env.device
        self.num_envs = env.num_envs
        self.use_delta_pose = False if "Rel" not in self.args_cli.task else True
        self.hand_side = "right" if self.add_right_hand else "left"

        self.target_object_name = f"{self.hand_side}_hand_object"
        self.demo_index = 0
        # if args_cli.save_path is not None:

        self.num_arm_actions = 6

        self.load_diffusion_model()
        super().__init__(
            env,
            env_cfg,
            args_cli,
            zarr_cfg=self.zarr_cfg,
        )

        self.temporal_action_buffer = TemporalEnsembleBufferAction(
            num_envs=self.env.num_envs,
            horizon_K=self.policy.horizon,
            action_dim=self.action_dim,
        )
        self.temporal_obs_buffer = TemporalEnsembleBufferObservation(
            num_envs=self.env.num_envs,
            horizon_K=self.policy.n_obs_steps,
            obs_dim=self.obs_dim,
        )
        self.temporal_image_buffer = TemporalEnsembleImageObservation(
            num_envs=self.env.num_envs,
            horizon_K=self.policy.n_obs_steps,
            obs_dim=self.obs_dim,
        )

        self.image_transform = T.Compose([
            T.Resize((self.image_dim[-2], self.image_dim[-1]),
                     interpolation=T.InterpolationMode.BICUBIC,
                     antialias=True),
            T.ToDtype(torch.float32, scale=True),  # replaces ToTensor in v2
        ])
        last_obs, _ = self.env.reset()

        # self.image_transform = get_image_transform(
        #     input_res=(last_obs["policy"]["rgb"].shape[-3:-1][::-1]),
        #     output_res=(self.image_dim[-2], self.image_dim[-1]),
        #     bgr_to_rgb=False)

    def load_diffusion_model(self):

        checkpoint = os.path.join(
            self.args_cli.diffusion_path, "checkpoints",
            f"{self.args_cli.diffusion_checkpoint}.ckpt")

        payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)

        cfg = payload['cfg']

        cfg.policy.num_inference_steps = 3
        cfg._target_ = "scripts.workflows.hand_manipulation.utils.diffusion.train_cfm_image_policy.TrainCFMUnetImageWorkspace"
        cls = hydra.utils.get_class(cfg._target_)

        workspace = cls(cfg, )
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        # get policy from workspace
        self.policy = workspace.model
        if cfg.training.use_ema:
            self.policy = workspace.ema_model

        device = torch.device(self.device)
        self.policy.to(device)
        self.policy.eval()

        self.chunk_size = self.policy.n_obs_steps
        self.obs_key = cfg.dataset.obs_key

        try:

            self.image_key = cfg.dataset.image_key
        except:
            self.image_key = ["image"]
        self.obs_dim = cfg.shape_meta.obs.agent_pos.shape[0]
        self.action_dim = cfg.shape_meta.action.shape[0]

        num_image = 0
        value = None
        for key in cfg.shape_meta.obs.keys():
            if "rgb" in key:
                num_image += 1
                value = cfg.shape_meta.obs[key]

        if value is None:
            value = torch.zeros((3, 256, 256))
            num_image = 1
        self.image_dim = [
            num_image,
            value.shape[0],
            value.shape[1],
            value.shape[2],
        ]
        self.zarr_cfg = cfg

    def get_demo_obs(self, index):
        obs_demo = []

        for key in self.obs_key:

            obs_demo.append(self.raw_data[key][index])
        obs_demo = torch.tensor(np.concatenate(obs_demo,
                                               axis=0)).to(self.device)

        self.temporal_obs_buffer.add_obs(index, obs_demo)

        image_demo = []
        for key in self.image_key:
            image_demo.append(self.raw_data[key][index][None])

        image_demo = torch.tensor(np.concatenate(image_demo,
                                                 axis=0)).to(self.device)
        self.temporal_image_buffer.add_obs(index, image_demo)

    def reset(self):
        self.env.reset()

        distractor_name = self.env_cfg["params"].get("distractor_name", [])
        init_rigid_object_pose = self.env.scene[
            f"{self.hand_side}_hand_object"]._data.reset_root_state
        if len(distractor_name) > 0:
            for distractor in distractor_name:

                dist = 100
                while True:

                    distractor_root_state = self.env.scene[
                        distractor]._data.reset_root_state
                    dist = torch.linalg.norm(distractor_root_state[..., :2] -
                                             init_rigid_object_pose[..., :2],
                                             dim=-1)
                    if dist > 0.20:
                        self.env.scene[
                            f"{self.hand_side}_hand_object"].write_root_pose_to_sim(
                                init_rigid_object_pose[..., :7],
                                env_ids=self.env_ids)
                        break

                    mdp.reset_root_state_uniform(
                        self.env,
                        self.env_ids,
                        pose_range=self.env_cfg["params"]["RigidObject"]
                        [distractor]["pose_range"],
                        velocity_range={},
                        asset_cfg=SceneEntityCfg(distractor))
        for i in range(20):
            if self.use_delta_pose:

                self.reset_robot_joints()
                actions = torch.zeros(self.env.action_space.shape,
                                      dtype=torch.float32,
                                      device=self.device)
                next_obs, rewards, terminated, time_outs, extras = self.env.step(
                    actions)
            else:
                if self.args_cli.real_eval_mode:

                    link7_pose = self.env.scene[
                        f"{self.hand_side}_hand"]._data.randomize_ee_pose[:, :
                                                                          7].clone(
                                                                          )
                else:
                    link7_pose = torch.as_tensor([[
                        0.500, -0.000, 0.500, 0.0, 9.2460e-01, -3.8094e-01, 0.0
                    ]]).to(self.device).repeat_interleave(self.env.num_envs,
                                                          dim=0)

                final_ee_pose = torch.cat([
                    link7_pose,
                    torch.zeros((self.env.num_envs, 16)).to(self.device)
                ],
                                          dim=-1)

                next_obs, rewards, terminated, time_outs, extras = self.env.step(
                    final_ee_pose)
        self.image_buffer = []

        return next_obs

    def open_loop_evaluate(self):

        self.reset_env()
        self.temporal_obs_buffer.reset(self.raw_data["actions"].shape[0],
                                       self.env.num_envs)

        self.temporal_image_buffer.reset(self.raw_data["actions"].shape[0],
                                         self.image_dim, self.env.num_envs)
        self.temporal_action_buffer.reset(self.raw_data["actions"].shape[0],
                                          self.env.num_envs)

        print("open_loop_evaluate")
        demo_action = self.raw_data["actions"]
        with torch.no_grad():

            for i in range(demo_action.shape[0]):
                self.get_demo_obs(i)
                obs_chunk = self.temporal_obs_buffer.compute_obs().clone()
                image_chunk = self.temporal_image_buffer.compute_obs().clone()

                rgb_dict = {}

                for num_image in range(image_chunk.shape[2]):
                    rgb_dict[f"rgb_{num_image}"] = image_chunk[:, :, num_image]

                obs_dict = {
                    "agent_pos": obs_chunk,
                } | rgb_dict
                predict_action = self.policy.predict_action(
                    obs_dict)["action_pred"]
                # for _ in range(predict_action.shape[1]):

                self.temporal_action_buffer.add_prediction(i, predict_action)
                # hand_action = self.temporal_action_buffer.compute_action()
                # self.env.step(predict_action[:, 0, :])
                self.env.step(
                    torch.as_tensor(demo_action[i]).to(
                        self.device).unsqueeze(0))
                # print(
                #     "mse",
                #     torch.nn.functional.mse_loss(
                #         predict_action[:, 0],
                #         torch.as_tensor(demo_action[i]).unsqueeze(0).to(
                #             self.device)))

        self.demo_index += 1
        return None

    def get_eval_obs(self, obs, index):
        obs_demo = []

        for key in self.obs_key:

            if key in ["joint_positions"]:
                obs_demo.append(obs[f"{self.hand_side}_hand_joint_pos"]
                                [:, :7].unsqueeze(1))
                continue

            obs_demo.append(obs[key])

        obs_demo = torch.cat(obs_demo, dim=1)
        # for key in self.obs_key:

        #     obs_demo.append(self.raw_data[key][index])
        # obs_demo = torch.tensor(np.concatenate(obs_demo,
        #                                        axis=0)).to(self.device)

        self.temporal_obs_buffer.add_obs(index, obs_demo)

        value = obs["rgb"]

        image_demo = []
        for image_index in range(value.shape[1]):
            resize_image = self.image_transform(value[:, image_index].permute(
                0, 3, 1, 2))

            image_demo.append(resize_image[:, None])

        image_demo = torch.cat(image_demo, dim=1).to(self.device)

        self.temporal_image_buffer.add_obs(index, image_demo)

        # old_image_demo = []
        # for key in self.image_key:

        #     resize_image = self.image_transform(
        #         torch.tensor(self.raw_data[key][index]))
        #     sharpened_batch = sharpen_batch(resize_image)

        #     old_image_demo.append(sharpened_batch[None])

        # old_image_demo = torch.cat(old_image_demo, dim=0).to(self.device)
        # viz_image_demo = np.concatenate(image_demo.permute(0, 2, 3,
        #                                                    1).cpu().numpy(),
        #                                 axis=1)
        # viz_old_image_demo = np.concatenate(old_image_demo.permute(
        #     0, 2, 3, 1).cpu().numpy(),
        #                                     axis=1)
        # plt.imshow(np.concatenate([viz_old_image_demo, viz_image_demo],
        #                           axis=0))
        # plt.axis("off")
        # plt.tight_layout()
        # plt.show()

        # self.temporal_image_buffer.add_obs(index, old_image_demo)

    def close_loop_evaluate(self):

        self.temporal_obs_buffer.reset(160, self.env.num_envs)
        self.temporal_action_buffer.reset(160, self.env.num_envs)
        self.temporal_image_buffer.reset(160, self.image_dim,
                                         self.env.num_envs)
        last_obs = self.reset()
        # last_obs = self.reset_env()
        self.image_buffer = []

        print("close_loop_evaluate")
        with torch.no_grad():

            for i in range(160):
                self.get_eval_obs(last_obs["policy"], i)
                # self.get_demo_obs(i)
                obs_chunk = self.temporal_obs_buffer.compute_obs().clone()
                image_chunk = self.temporal_image_buffer.compute_obs().clone()

                rgb_dict = {}

                rgb_dict[self.image_key[0]] = image_chunk[:, :, 0]

                obs_dict = {
                    "agent_pos": obs_chunk,
                } | rgb_dict

                predict_action = self.policy.predict_action(
                    obs_dict)["action_pred"]

                self.temporal_action_buffer.add_prediction(i, predict_action)
                # hand_action = self.temporal_action_buffer.compute_action()
                # next_obs, rewards, terminated, time_outs, extras = self.env.step(
                #     hand_action)
                next_obs, rewards, terminated, time_outs, extras = self.env.step(
                    predict_action[:, 0])

                self.image_buffer.append(
                    last_obs["policy"]["rgb"][:, 0].cpu().numpy())

                last_obs = copy.deepcopy(next_obs)
        self.demo_index += 1
        return self.env.scene[
            f"{self.hand_side}_hand_object"].data.root_state_w[..., 2] > 0.3

    def close_loop_evaluate2(self):

        last_obs = self.reset()
        self.last_diffusion_obs = self.get_diffusion_obs(last_obs["policy"])

        # last_obs = self.reset_env()
        self.image_buffer = []

        print("close_loop_evaluate")
        with torch.no_grad():

            for i in range(160):

                predict_action = self.policy.predict_action(
                    self.last_diffusion_obs)["action_pred"]

                last_obs, rewards, terminated, time_outs, extras = self.env.step(
                    predict_action[:, 0])
                self.last_diffusion_obs = self.get_diffusion_obs(
                    last_obs["policy"])

        self.demo_index += 1

        object_pose = self.env.scene[
            f"{self.hand_side}_hand_object"]._data.root_state_w[:, :3].clone()
        object_pose[:, :3] -= self.env.scene.env_origins
        lift_or_not = (object_pose[:, 2] > 0.30)
        overhigh_or_not = (object_pose[:, 2] < 0.60)
        outofbox_or_not = ((object_pose[:, 0] < 0.65) &
                           (object_pose[:, 0] > 0.3) &
                           (object_pose[:, 1] < 0.3) &
                           (object_pose[:, 1] > -0.3))
        success = lift_or_not & overhigh_or_not & outofbox_or_not
        return success
