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
from scripts.workflows.bimanual.utils.wandb_utils import log_media
import gc

sys.path.append(".")
from scripts.workflows.bimanual.utils.trajectory_generator import TrajectoryGenerator

from scripts.workflows.bimanual.utils.ControlEnv import ControlEnv
from scripts.workflows.bimanual.utils.object_env import ObjectEnv
from tools.curobo_planner import IKPlanner


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
        self.init_settings()

        TrajectoryGenerator.__init__(self)
        ControlEnv.__init__(self)
        ObjectEnv.__init__(self)

        self.init_callback()

    def init_gripper(self):

        deform_object_cfg = self.env.scene["deform_object"].cfg.deform_cfg
        floating_gripper_setting = deform_object_cfg["floating_gripper"]

        self.static_frames = floating_gripper_setting["static_frames"]

        self.gripper_offset = torch.as_tensor(
            floating_gripper_setting["gripper_offset"]).to(self.device)

        self.random_orientation_range = floating_gripper_setting[
            "random_orientation_range"]

        self.trajectories_dir = torch.as_tensor(
            floating_gripper_setting["random_position_dir"]).to(self.device)
        self.gripper_actions = torch.as_tensor(
            [floating_gripper_setting["gripper_actions"]]).to(self.device)

    def init_robot(self):

        robot_cfg = self.env.scene["robot"].cfg.articulation_cfg[
            "robot_setting"]
        self.end_frame = robot_cfg["motion_type"]["push"]["end_frame"]
        self.static_frame = robot_cfg["motion_type"]["push"]["static_frame"]

        self.init_ee_pose = torch.as_tensor([
            settings["init_pose"]
            for settings in robot_cfg["motion_type"].values()
        ]).to(self.device)
        self.ee_motion_vel = torch.as_tensor([
            settings["vel"] for settings in robot_cfg["motion_type"].values()
        ]).to(self.device)
        self.key_motions = list(robot_cfg["motion_type"].keys())

        self.reset_robot_target_pose = self.env.scene[
            "robot"]._data.default_joint_pos[:, :9].clone()
        self.reset_robot_target_pose[:, -2:] = -1

        self.gripper_actions = torch.as_tensor([-1]).to(self.device)

        self.random_orientation_range = robot_cfg["motion_type"]["push"][
            "random_orientation_rang"]

        if "gripper" in self.env.scene.keys():

            init_pos = torch.as_tensor([0.3, 0.0, 0.5, 0.0, 1.0, 0.0,
                                        0.0]).to(self.device)

            curobo_ik = IKPlanner()
            self.frozen_robot_sol = curobo_ik.plan_motion(
                init_pos[:3].unsqueeze(0), init_pos[3:7].unsqueeze(0))
            self.reset_robot_target_pose = self.env.scene[
                "robot"]._data.default_joint_pos[:, :9].clone()
            self.reset_robot_target_pose[:, -2:] = -1

    def init_settings(self):
        deform_object_cfg = self.env.scene["deform_object"].cfg.deform_cfg

        self.include_robot = False
        self.has_gripper = False

        self.explore_type = "target"
        self.explore_action_index = 0
        self.num_gripper_actions = 0
        self.gripper_object = []

        self.num_gripper = 0

        for object_name in self.env.scene.keys():
            if "gripper" in object_name:
                self.has_gripper = True
                self.gripper_object.append(self.env.scene[object_name])
                self.num_gripper = len(self.gripper_object)

            if "robot" in object_name:
                self.init_robot()
                self.include_robot = True

        if self.has_gripper:
            self.init_gripper()
        if self.include_robot:
            self.init_robot()
        self.num_explore_actions = deform_object_cfg["env_setting"][
            "num_explore_actions"]
        self.num_robot_actions = deform_object_cfg["env_setting"][
            "num_robot_actions"] if self.include_robot else 0

        self.num_gripper_actions = deform_object_cfg["env_setting"][
            "num_gripper_actions"] if self.has_gripper else 0

    def reset_episode(self,
                      explore_type,
                      explore_action_index,
                      reset_gripper=True):

        super().new_episode_training_start(explore_type, explore_action_index,
                                           reset_gripper)
        self.reset_deformable_visual_obs(boolen=True)

        self.explore_type = explore_type
        self.explore_action_index = explore_action_index

        if explore_type == "target":

            cur_attachment_points = self.get_sample_attachment_points(
                explore_type, explore_action_index)

            self.generate_trajectories(cur_attachment_points, )

        next_obs = self.env_reset(explore_type, reset_gripper,
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
                                   len(self.target_attachment_points))
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
                    next_obs = self.reset_episode(
                        explore_type,
                        explore_action_index=k,
                        reset_gripper=True
                        if rollout_count == 0 and self.has_gripper else False)
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
