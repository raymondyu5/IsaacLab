import torch
import isaaclab.utils.math as math_utils
import sys
from typing import Union
from pxr import Sdf, Usd
import weakref
import warnings
import logging
import gym

from isaaclab_tasks.utils import parse_env_cfg
from scripts.workflows.sysID.ASID.tool.utilis import save_target_video
from scripts.workflows.sysID.utils.wandb_utils import log_media
import gc

sys.path.append(".")
from scripts.workflows.sysID.utils.trajectory_generator import TrajectoryGenerator

from scripts.workflows.sysID.utils.ControlEnv import ControlEnv
from scripts.workflows.sysID.utils.object_env import ObjectEnv


class GripperActionManager(ControlEnv, ObjectEnv, TrajectoryGenerator):

    def __init__(
        self,
        env,
        task,
        buffer=None,
        use_wandb=False,
        require_segmentation=True,
        reset_camera_obs_list=None,
        render_all=False,
        debug_vis=False,
    ):
        if reset_camera_obs_list is None:
            reset_camera_obs_list = ["seg_rgb", "whole_pc", "seg_pc"]

        self.env = weakref.proxy(env)
        self.num_envs = env.num_envs
        self.device = env.device
        self.task = task
        self.reset_camera_obs_list = reset_camera_obs_list
        self.render_all = render_all
        self.debug_vis = debug_vis
        self.require_segmentation = require_segmentation
        self.num_explore_actions = self.env.scene[
            "deform_object"].cfg.deform_cfg["env_setting"][
                "num_explore_actions"]
        self.use_wandb = use_wandb

        self.buffer = buffer

        self.trajectory_generator = TrajectoryGenerator(
            env, task, self.num_explore_actions)

        self.include_robot = "Abs" in self.task or "Rel" in self.task
        self.has_gripper = True if "gripper" in self.env.scene.keys(
        ) else False

        ControlEnv.__init__(self)
        ObjectEnv.__init__(self)

        self.init_callback()

    def new_episode_training_start(self,
                                   explore_type,
                                   explore_action_index,
                                   reset_gripper=True):

        super().new_episode_training_start(explore_type, explore_action_index,
                                           reset_gripper)
        self.reset_deformable_visual_obs(boolen=True)
        self.trajectory_generator.sample_attachment_points = self.get_sample_attachment_points(
            explore_type, explore_action_index)

        self.explore_type = explore_type
        self.explore_action_index = explore_action_index

        self.trajectory_generator.on_training_start()

        next_obs = self.technical_reset(explore_type, reset_gripper,
                                        explore_action_index)

        return next_obs

    def __del__(self):
        self._handle = None

    def step_gripper_manager(self,
                             sample_parms=None,
                             explore_type=None,
                             log_path=None,
                             num_loop=None,
                             rollout_num=1,
                             save_interval=9):

        if explore_type == "train":
            num_interactions = min(self.num_explore_actions,
                                   len(self.sample_attachment_points))
        else:
            num_interactions = 1

        for k in range(num_interactions):
            print('Interaction:', k)
            transition = {}

            for rollout_count in range(rollout_num):

                print("Rollout count", rollout_count)

                self.randomize_deformable_properties(
                    random_method="customize",
                    sample_parms=sample_parms[rollout_count *
                                              self.env.num_envs:
                                              (rollout_count + 1) *
                                              self.env.num_envs])
                while True:
                    next_obs = self.new_episode_training_start(
                        explore_type,
                        explore_action_index=k,
                        reset_gripper=True if rollout_count == 0 else False)
                    images_buffer, success = self.step_env(
                        k,
                        next_obs,
                        explore_type,
                        save_interval,
                        transition,
                    )
                    if success:
                        break
                    else:
                        transition = {}

            if self.buffer is not None:
                self.buffer.create_transitions(transition,
                                               cache_type=explore_type)
        # Store and clear buffer after all interactions

        if self.buffer is not None:
            self.buffer._store_transition(cache_type=explore_type)
            self.buffer._clear_cache(cache_type=explore_type)

        # Log images and media if required
        if self.use_wandb and explore_type in ["target", "eval"]:
            self.log_videos_and_media(images_buffer, log_path, num_loop,
                                      explore_type)

        # Clean up
        del images_buffer
        gc.collect()
        torch.cuda.empty_cache()

    def collect_images(self, next_obs):
        """Helper function to collect images from the observation."""
        images = []
        if self.require_segmentation and "seg_rgb" in next_obs["policy"]:
            images.append(next_obs["policy"]["seg_rgb"].cpu())
        elif "rgb" in next_obs["policy"]:
            images.append(next_obs["policy"]["rgb"].cpu())
        return images

    def log_videos_and_media(self, images_buffer, log_path, num_loop,
                             explore_type):
        """Helper function to log videos and media using wandb."""
        if len(images_buffer) > 5:
            save_target_video(images_buffer,
                              log_path,
                              num_loop,
                              folder_name=f"{explore_type}/video",
                              num_explore_actions=self.num_explore_actions)
        log_media(self.num_explore_actions,
                  self.buffer,
                  log_path,
                  num_loop,
                  log_type=explore_type)


def initialize_gripper(
    env,
    args_cli,
    buffer=None,
    use_wandb=False,
    require_segmentation=True,
    reset_camera_obs_list=None,
    render_all=False,
    debug_vis=False,
):

    self = GripperActionManager(
        env,
        args_cli.task,
        buffer=buffer,
        use_wandb=use_wandb,
        require_segmentation=require_segmentation,
        reset_camera_obs_list=reset_camera_obs_list,
        render_all=render_all,
        debug_vis=debug_vis,
    )

    return self


# if target_gripper_traj is not None:
#     # float_gripper_abs_actions = self.assemble_abs_gripper_actions(
#     #     target_gripper_traj, gripper_actions)
#     # actions = self.assemble_abs_robot_abs_gripper_actions(
#     #     float_gripper_abs_actions, )
#     # robot_actions = actions[:, :7]
#     # gripper_action = actions[:, 7:9]

#     # self.env.scene["robot"].root_physx_view.set_dof_positions(
#     #     torch.cat([
#     #         self.env.scene["robot"]._data.default_joint_pos[:, :7],
#     #         gripper_action
#     #     ],
#     #               dim=-1), indices)
#     # self.env.sim.step(render=True)

#     robot_actions = self.env.scene["gripper"].data.default_root_state
#     robot_actions[:, 3:7] = math_utils.quat_mul(
#         robot_actions[:, 3:7],
#         self.env.scene["robot"].data.default_ee_pose[:, 3:7])

#     robot_jpos = self.ik_planner.plan_motion(
#         robot_actions[:, :3], robot_actions[:, 3:7],
#         self.env.scene["robot"]._data.default_joint_pos)
