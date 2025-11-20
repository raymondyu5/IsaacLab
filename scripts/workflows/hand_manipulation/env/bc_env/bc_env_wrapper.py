from scripts.workflows.hand_manipulation.env.rl_env.replay_rl_wrapper import ReplayRLWrapper
from scripts.workflows.hand_manipulation.env.bc_env.bc_diffusion_wrapper import BCDiffusionWrapper
from scripts.workflows.hand_manipulation.env.bc_env.image_bc_diffusion_wrapper import ImageBCDiffusionWrapper
from scripts.workflows.hand_manipulation.env.bc_env.pcd_bc_diffusion_wrapper import PCDBCDiffusionWrapper

from scripts.workflows.hand_manipulation.env.bc_env.state_bc_diffusion_wrapper import StateBCDiffusionWrapper
import torch

from scripts.workflows.utils.multi_datawrapper import MultiDatawrapper
from scripts.workflows.open_policy.utils.buffer_utils import reset_buffer, update_buffer, filter_out_data

from tools.visualization_utils import vis_pc, visualize_pcd
import isaaclab.utils.math as math_utils
from isaaclab.utils.math import LPFilter
import time
from tools.visualization_utils import *

from matplotlib import pyplot as plt


class HandBCEnvWrapper:

    def __init__(
        self,
        env,
        env_config,
        args_cli,
        use_relative_pose=False,
    ):
        self.env = env
        self.device = self.env.unwrapped.device
        self.args_cli = args_cli

        self.use_relative_pose = use_relative_pose
        self.env_config = env_config
        self.action_space = self.env.unwrapped.action_space
        self.observation_space = self.env.unwrapped.observation_space
        self.hand_side = "right" if args_cli.add_right_hand else "left"
        self.finger_lpfilter = LPFilter(alpha=1.0, max_step=0.2)

        if args_cli.eval_mode == "replay":

            self.replay_env = ReplayRLWrapper(
                env,
                env_config,
                args_cli,
            )

            self.step = self.replay_env.replay
            return

        if args_cli.action_framework in [
                "diffusion", "image_diffusion", "pcd_diffusion",
                "state_diffusion"
        ]:
            if args_cli.action_framework == "diffusion":

                self.diffusion_env = BCDiffusionWrapper(
                    env,
                    env_config,
                    args_cli,
                )
            elif args_cli.action_framework == "image_diffusion":

                self.diffusion_env = ImageBCDiffusionWrapper(
                    env,
                    env_config,
                    args_cli,
                )
            elif args_cli.action_framework == "pcd_diffusion":

                self.diffusion_env = PCDBCDiffusionWrapper(
                    env,
                    env_config,
                    args_cli,
                )
            elif args_cli.action_framework == "state_diffusion":

                self.diffusion_env = StateBCDiffusionWrapper(
                    env,
                    env_config,
                    args_cli,
                )
            if args_cli.eval_mode == "open_loop":
                self.step = self.diffusion_env.open_loop_evaluate
            elif args_cli.eval_mode == "close_loop":
                self.step = self.diffusion_env.close_loop_evaluate

            if args_cli.analysis:

                self.step = self.diffusion_env.analyze_evaluation

            setattr(self.diffusion_env, "get_diffusion_obs",
                    self.get_diffusion_obs)
            self.action_framework = args_cli.action_framework
            setattr(self.diffusion_env, "evaluate_success",
                    self.evaluate_success)
            setattr(self.diffusion_env, "finger_lpfilter",
                    self.finger_lpfilter)

            if self.args_cli.save_path is not None:
                filter_keys = []
                if self.args_cli.analysis:
                    filter_keys = ['seg_pc']

                self.collector_interface = MultiDatawrapper(
                    self.args_cli,
                    self.env_config,
                    save_path=self.args_cli.save_path,
                    filter_keys=filter_keys,
                    save_zarr=True)
                setattr(self.diffusion_env, "collector_interface",
                        self.collector_interface)
                reset_buffer(self)
                setattr(self.diffusion_env, "add_data", self.add_data)
            else:
                setattr(self.diffusion_env, "collector_interface", None)

    def add_data(self, next_obs, last_obs, actions, rewards, done):

        self.obs_buffer.append(last_obs)

        self.action_buffer.append(actions.cpu())
        self.rewards_buffer.append(rewards)

        self.does_buffer.append(done)

    def lift_or_not(self, success_flag):

        if success_flag.sum() > 0 or self.args_cli.use_failure:
            if self.args_cli.save_path is not None:

                if self.args_cli.use_failure:

                    success_index = torch.nonzero(success_flag,
                                                  as_tuple=True)[0]
                    failure_index = torch.nonzero(~success_flag,
                                                  as_tuple=True)[0]
                    total_data = min([
                        len(failure_index) /
                        float(self.args_cli.failure_ratio), self.env.num_envs
                    ])
                    # Step 4: concatenate final indices
                    num_fail = int(total_data *
                                   float(self.args_cli.failure_ratio))
                    num_succ = int(total_data - num_fail)

                    if num_fail <= len(failure_index):
                        chosen_fail = failure_index[torch.randperm(
                            len(failure_index))[:num_fail]]
                    else:
                        # not enough failures, so just take all of them
                        chosen_fail = failure_index

                    # Successes
                    if num_succ <= len(success_index):
                        chosen_succ = success_index[torch.randperm(
                            len(success_index))[:num_succ]]
                    else:
                        chosen_succ = success_index

                    # final index
                    final_index = torch.cat([chosen_fail, chosen_succ]).cpu()

                    filter_out_data(self, final_index)
                else:

                    index = torch.nonzero(success_flag, as_tuple=True)[0]

                    filter_out_data(self, index.cpu())

        return success_flag

    def save_data_to_buffer(self, success):

        filter_out_data(self, torch.where(success)[0].tolist())

    def evaluate_success(self, object):

        object_pose = self.env.unwrapped.scene[
            f"{self.hand_side}_hand_object"]._data.root_state_w[:, :3].clone()
        object_pose[:, :3] -= self.env.unwrapped.scene.env_origins

        if "Place" in self.args_cli.task:
            target_object_state = self.env.scene[
                f"{self.hand_side}_hand_place_object"].data.root_state_w[
                    ..., :7]
            pick_object_state = self.env.scene[
                f"{self.hand_side}_hand_object"].data.root_state_w[..., :7]
            success = torch.linalg.norm(target_object_state[:, :2] -
                                        pick_object_state[:, :2],
                                        dim=-1) < 0.15

        else:
            lift_or_not = (object_pose[:, 2] > 0.20)
            overhigh_or_not = (object_pose[:, 2] < 0.65)
            outofbox_or_not = ((object_pose[:, 0] < 0.65) &
                               (object_pose[:, 0] > 0.3) &
                               (object_pose[:, 1] < 0.3) &
                               (object_pose[:, 1] > -0.3))
            success = lift_or_not  # & overhigh_or_not & outofbox_or_not

        if success.sum() > 0 or self.args_cli.use_failure:
            if self.args_cli.save_path is not None:

                if self.args_cli.use_failure:
                    final_index = torch.arange(self.env.num_envs).cpu()

                    filter_out_data(object, final_index)
                else:

                    index = torch.nonzero(success, as_tuple=True)[0]

                    filter_out_data(object, index.cpu())

        # import pdb
        # pdb.set_trace()
        # visualize_pcd([vis_pc(self.obs_buffer[0]["policy"]["seg_pc"][0].cpu().numpy()[0])])

        # plt.imshow(self.obs_buffer[0]["policy"]["rgb"][0][0].cpu().numpy())
        # plt.show()
        reset_buffer(self)
        return success

    def get_diffusion_obs(self, obs):

        obs_demo = []

        for key in self.diffusion_env.obs_key:

            obs_demo.append(obs[key])

        obs_demo = torch.cat(obs_demo, dim=1)

        if self.action_framework in ["pcd_diffusion", "image_diffusion"]:

            image_demo = []

            for key in self.diffusion_env.image_key:
                if self.action_framework in ["pcd_diffusion"]:

                    for key in self.diffusion_env.image_key:

                        seg_pcd = obs[key]
                        sampled_pcd = []
                        for pcd in seg_pcd:

                            # sampled_pcd.append(
                            #     math_utils.fps_points(
                            #         pcd, self.diffusion_env.image_dim[-1]))

                            points_index = torch.randperm(pcd.shape[-2]).to(
                                self.device)

                            sampled_pcd.append(
                                pcd[:, points_index[:self.diffusion_env.
                                                    image_dim[-1]]])
                    sampled_pcd = torch.cat(sampled_pcd, dim=0)
                    image_demo.append(sampled_pcd)

                    image_demo = torch.cat(image_demo, dim=0).permute(0, 2, 1)

                    obs_dict = {
                        "agent_pos": obs_demo.unsqueeze(1),
                        "seg_pc": image_demo.unsqueeze(1),
                    }
                else:

                    transfomred_image = self.diffusion_env.image_transform(
                        obs["rgb"][:, 0].permute(0, 3, 1, 2))
                    image_demo.append(transfomred_image)
                    image_demo = torch.cat(image_demo, dim=0)
                    obs_dict = {
                        "agent_pos": obs_demo.unsqueeze(1),
                        self.diffusion_env.image_key[0]:
                        image_demo.unsqueeze(1),
                    }

        elif self.action_framework == "state_diffusion":
            obs_dict = {
                "obs": obs_demo.unsqueeze(1),
            }

        return obs_dict
