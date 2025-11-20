import torch
import numpy as np

import copy

import imageio

from scripts.workflows.open_policy.utils.buffer_utils import reset_buffer, update_buffer
import tqdm

from scripts.workflows.hand_manipulation.env.rl_env.rl_step_wrapper import RLStepWrapper
from scripts.workflows.hand_manipulation.env.rl_env.rl_reactive_vae_step_wrapper import RLReactiveVAEStepWrapper
from scripts.workflows.hand_manipulation.env.rl_env.rl_cfm_wrapper import RLCFMStepWrapper
from scripts.workflows.hand_manipulation.env.rl_env.rl_bc_wrapper import BCReplayDatawrapper

from scripts.workflows.hand_manipulation.env.rl_env.eval_rl_wrapper import EvalRLWrapper
from scripts.workflows.hand_manipulation.env.rl_env.collect_rl_wrapper import CollectRLWrapper
from scripts.workflows.hand_manipulation.env.rl_env.replay_rl_wrapper import ReplayRLWrapper
from scripts.workflows.open_policy.utils.buffer_utils import reset_buffer, update_buffer, filter_out_data
import gymnasium as gym


class RLDatawrapperEnv:

    def __init__(self,
                 env,
                 env_config,
                 args_cli,
                 use_relative_pose=False,
                 use_joint_pose=False,
                 eval_mode=False,
                 replay_mode=False,
                 collect_mode=False):

        self.env = env

        self.args_cli = args_cli

        self.use_relative_pose = use_relative_pose
        self.use_joint_pose = use_joint_pose
        self.env_config = env_config
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.task = "place" if "Place" in args_cli.task else "grasp"

        if replay_mode:

            self.replay_env = ReplayRLWrapper(
                env,
                env_config,
                args_cli,
            )
            self.replay = self.replay_env.replay
            return

        if self.args_cli.action_framework in [
                "pca", "vae"
        ] or self.args_cli.action_framework is None:
            self.wrapper = RLStepWrapper(args_cli, env_config, env)
        elif self.args_cli.action_framework in ["diffusion"]:

            self.wrapper = RLCFMStepWrapper(args_cli, env_config, env)
        elif self.args_cli.action_framework in ["reactive_vae"]:
            self.wrapper = RLReactiveVAEStepWrapper(
                args_cli,
                env_config,
                env,
            )
        self.step = self.wrapper.step
        self.reset = self.wrapper.reset
        self.num_hand_joints = self.wrapper.num_hand_joints
        self.num_finger_actions = self.wrapper.num_finger_actions
        self.hand_side = self.wrapper.hand_side
        self.num_arm_actions = self.wrapper.num_arm_actions

        setattr(self.wrapper, "eval_success", self.eval_success)
        setattr(self.wrapper, "eval_mode", eval_mode)
        if self.args_cli.rl_type in ["ppobc"]:

            bc_wrapper = BCReplayDatawrapper(
                self.env,
                self.env_config,
                args_cli,
            )
            bc_wrapper.step()
            self.bc_rollout_buffer = bc_wrapper.rollout_buffer
        else:
            self.bc_rollout_buffer = None

        if eval_mode:

            if collect_mode:
                self.eval_env = CollectRLWrapper(
                    env,
                    env_config,
                    args_cli,
                    wrapper=self.wrapper,
                    use_relative_pose=use_relative_pose,
                    use_joint_pose=self.use_joint_pose,
                    hand_side=self.wrapper.hand_side)

            else:
                self.eval_env = EvalRLWrapper(
                    env,
                    env_config,
                    args_cli,
                    use_relative_pose=use_relative_pose,
                    use_joint_pose=self.use_joint_pose,
                    hand_side=self.wrapper.hand_side)

                self.init_eval_result_folder = self.eval_env.init_eval_result_folder

                self.eval_checkpoint = self.eval_env.eval_checkpoint

                if self.args_cli.eval_disturbance:
                    self.eval_checkpoint = self.eval_env.eval_disturbance
                self.eval_all_checkpoint = self.eval_env.eval_all_checkpoint
                setattr(self.eval_env, "reset", self.wrapper.reset)
                setattr(self.eval_env, "step", self.wrapper.step)
            setattr(self.eval_env, "num_hand_joints",
                    self.wrapper.num_hand_joints)
            setattr(self.eval_env, "num_finger_actions",
                    self.wrapper.num_finger_actions)

            setattr(self.eval_env, "num_arm_actions",
                    self.wrapper.num_arm_actions)

    def eval_success(self, last_obs, task=None):
        if task is not None:
            task = task

        else:
            task = self.task

        if task == "grasp":
            success = (last_obs["policy"]
                       [f"{self.hand_side}_manipulated_object_pose"][:,
                                                                     2]) > 0.3
        elif task == "place":

            pick_object_state = (
                last_obs["policy"][f"{self.hand_side}_manipulated_object_pose"]
                [:, :3])
            place_target_state = (
                last_obs["policy"][f"{self.hand_side}_hand_place_object_pose"]
                [:, :3])
            success = torch.linalg.norm(pick_object_state[:, :2] -
                                        place_target_state[:, :2],
                                        dim=1) < 0.10
        return success
