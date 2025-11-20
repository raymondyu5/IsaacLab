from scripts.workflows.hand_manipulation.utils.cloudxr.franka_teleoperation.se3_franka_wrapper import Se3FrankaWrapper
from isaaclab_tasks.manager_based.manipulation.lift.utils.config_cluster_rigids import randomize_object_pose

import torch
import random

from isaaclab.managers import SceneEntityCfg


class TeleopEnvWrapper:

    def __init__(
        self,
        env,
        env_cfg,
        args_cli,
    ):
        self.teleop_env = Se3FrankaWrapper(
            env,
            env_cfg,
            args_cli,
        )
        self.args_cli = args_cli

        setattr(self.teleop_env, "reset_task_env", self.reset_task_env)
        self.env_ids = torch.arange(self.teleop_env.env.num_envs,
                                    device=self.teleop_env.device)
        self.init_settings()

        self.teleop_env.reset_teleoperation()

    def init_settings(self):
        if "lift" in self.args_cli.task.lower():
            self.pick_object_range = self.teleop_env.env_cfg["params"][
                "pick_object_range"]
            self.pick_object_list = self.teleop_env.env_cfg["params"][
                "pick_object_list"]
            self.place_object_range = self.teleop_env.env_cfg["params"][
                "place_object_range"]
            self.place_object_list = self.teleop_env.env_cfg["params"][
                "place_object_list"]
            self.min_separation = self.teleop_env.env_cfg["params"][
                "min_separation"]

    def reset_task_env(self):

        if "lift" in self.args_cli.task.lower():
            target_pick_object = random.choice(self.pick_object_list)
            target_place_object = random.choice(self.place_object_list)
            reset_pick_height = self.teleop_env.env_cfg["params"]
            reset_pick_height = self.teleop_env.env_cfg["params"][
                "RigidObject"][target_pick_object]["pos"][2]
            reset_place_height = self.teleop_env.env_cfg["params"][
                "RigidObject"][target_place_object]["pos"][2]
            randomize_object_pose(
                self.teleop_env.env,
                self.env_ids,
                [
                    SceneEntityCfg(target_pick_object),
                    SceneEntityCfg(target_place_object)
                ],
                self.min_separation,
                self.pick_object_range,
                self.place_object_range,
                reset_height=[[reset_pick_height], [reset_place_height]],
                max_sample_tries=50,
            )

    def step(self):
        reset = self.teleop_env.step()
