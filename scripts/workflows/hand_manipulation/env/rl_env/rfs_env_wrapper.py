from scripts.workflows.hand_manipulation.env.rl_env.rl_wrapper import RLDatawrapperEnv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym

from gymnasium import spaces


def load_dinov3(name='dinov3_vits16'):
    """
    Load dinov3 model
    """

    REPO_DIR = "submodule/dinov3"  # your local repo (with hubconf.py)
    if name == 'dinov3_vits16':

        CKPT_PATH = "submodule/dinov3/ckpt/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

    model = torch.hub.load(repo_or_dir=REPO_DIR,
                           model="dinov3_vits16",
                           source="local",
                           weights=CKPT_PATH).cuda()
    return model


class RFSEnvWrapper(RLDatawrapperEnv):

    def __init__(
        self,
        env,
        env_config,
        args_cli,
        use_relative_pose=False,
        use_joint_pose=False,
        eval_mode=False,
        replay_mode=False,
    ):
        super().__init__(
            env,
            env_config,
            args_cli,
            use_relative_pose,
            use_joint_pose,
            eval_mode,
            replay_mode,
        )
        del self.step, self.reset
        self.use_cam_obs = self.env_config["params"]["Camera"]["initial"]

        if self.use_cam_obs:
            self.image_model = load_dinov3()

            reset_obs, _ = self.wrapper.reset()
            
            with torch.no_grad():
                image_feature = self.image_model(
                    reset_obs["policy"]["rgb"].squeeze(1).permute(0, 3, 1, 2).to(
                        torch.float32) / 255.0)

            old_space = self.env.unwrapped.observation_space.spaces["policy"]

            # Rebuild without "rgb"
            new_dict = {
                k: v
                for k, v in old_space.items()
                if k in [f"{self.hand_side}_hand_joint_pos"]
            }
            new_dict["image_feature"] = gym.spaces.Box(
                low=-10.0,
                high=10.0,
                shape=image_feature.shape,
                dtype=np.float32,
            )

            # Replace it
            self.env.unwrapped.observation_space.spaces["policy"] = spaces.Dict(
                new_dict)

    def step(self, actions):
        rollouts = self.wrapper.step(actions)
        if self.use_cam_obs:
            new_obs = self.filterout_obs(rollouts[0]["policy"])

            return (new_obs, ) + rollouts[1:]
        return rollouts

    def reset(self):
        rollouts = self.wrapper.reset()

        if self.use_cam_obs:
            new_obs = self.filterout_obs(rollouts[0]["policy"])

            return (new_obs, ) + rollouts[1:]
        return rollouts

    def filterout_obs(self, obs):
        filtered_obs = dict()
        for key in obs.keys():
            if key in [f"{self.hand_side}_hand_joint_pos"]:
                filtered_obs[key] = obs[key]
        with torch.no_grad():

            image_feature = self.image_model(
                obs["rgb"].squeeze(1).permute(0, 3, 1, 2).to(torch.float32) /
                255.0)
            filtered_obs["image_feature"] = image_feature
        return {"policy": filtered_obs}
