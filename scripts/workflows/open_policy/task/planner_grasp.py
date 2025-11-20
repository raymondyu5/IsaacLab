from scripts.workflows.vlm_failure.stack_block.task.env_grasp import GrasperEnv
from scripts.workflows.automatic_articulation.task.env_placement import PlacementEnv
from tools.curobo_planner import MotionPlanner
import torch
from scripts.workflows.automatic_articulation.utils.process_action import process_action, curobo2robot_actions
import isaaclab.utils.math as math_utils
from scripts.workflows.utils.multi_datawrapper import MultiDatawrapper
import isaaclab.envs.mdp as mdp
from scripts.workflows.open_policy.utils.buffer_utils import reset_buffer, update_buffer
import copy
from matplotlib import pyplot as plt
import numpy as np

from scripts.workflows.open_policy.utils.criterion import criterion_pick_place
from source.isaaclab.isaaclab.envs.mdp.events import reset_rigid_articulation


class PlannerGrasp:

    def __init__(self, env, args_cli, env_config):
        self.env = env
        self.device = env.device

        self.args_cli = args_cli
        self.env_config = env_config
        self.env_ids = torch.arange(self.env.num_envs, device=self.device)
        self.init_planner()
        self.grasp_env = GrasperEnv(env,
                                    self.planner,
                                    collision_checker=False,
                                    env_config=self.env_config)
        self.use_relative_pose = True if "Rel" in args_cli.task else False
        if args_cli.save_path is not None:
            self.collector_interface = MultiDatawrapper(
                args_cli,
                env_config,
                filter_keys=["gs_image"],
                load_path=None,
                save_path=args_cli.save_path,
                use_fps=False,
                use_joint_pos=False if "joint" not in args_cli.task else True,
            )
            self.collector_interface.init_collector_interface()
            reset_buffer(self)

        self.target_object_name = self.env_config["params"]["Task"][
            "target_object"]
        self.placement_object_name = self.env_config["params"]["Task"][
            "placement"]["placement_object"]
        self.bbox_region = self.env_config["params"]["Task"][
            "success_condition"]["bbox_region"]

    def init_planner(self):
        self.planner = MotionPlanner(
            self.env,
            collision_checker=False,
            reference_prim_path="/World/envs/env_0/Robot",
            ignore_substring=[
                "/World/envs/env_0/Robot",
                "/World/GroundPlane",
                "/World/collisions",
                "/World/light",
                "/curobo",
            ],
        )
        self.open_gripper_horizon = self.env_config["params"]["Task"][
            "placement"]["open_gripper_horizon"]
        self.placement_height = self.env_config["params"]["Task"]["placement"][
            "placement_height"]

    def reset(self):
        if self.args_cli.grasp_mode is not None:
            self.grasp_env.grasp_mode = "cube"
            self.grasp_env.grasp_object = self.env.scene[
                self.env_config["params"]["Task"]["target_object"]]

        last_obs, _ = self.env.reset()

        for reset_object_name in self.env_config["params"]["Task"][
                "reset_object_names"]:
            if reset_object_name not in self.env_config["params"]["RL_Train"][
                    "rigid_object_names"]:
                continue

            mdp.reset_rigid_articulation(
                self.env,
                self.env_ids,
                target_name=reset_object_name,
                pose_range=self.env_config["params"]["RigidObject"]
                [reset_object_name]["pose_range"])

        for i in range(20):

            reset_joint_pose = self.env.scene[
                "robot"].root_physx_view.get_dof_positions().clone()
            reset_joint_pose[:, :self.grasp_env.init_jpos.
                             shape[-1]] = self.grasp_env.init_jpos[:, 0]
            self.env.scene["robot"].root_physx_view.set_dof_positions(
                reset_joint_pose, self.env_ids)
            if self.use_relative_pose:
                actions = torch.zeros(self.env.action_space.shape,
                                      device=self.device)

                last_obs, reward, terminate, time_out, info = self.env.step(
                    actions)
            else:
                last_obs, reward, terminate, time_out, info = self.env.step(
                    self.grasp_env.init_ee_pose)

        self.grasp_env.sample_gripper_pose(last_obs)

        self.grasp_env.grasp_object_name = self.env_config["params"]["Task"][
            "target_object"]
        self.grasp_env.grasp_object = self.env.scene[
            self.grasp_env.grasp_object_name]

        self.grasp_env.reset(last_obs, resample=True)

        return last_obs

    def sample_placement_pose(self, last_obs, target_pos=None):
        self.placement_object = self.env.scene[
            self.env_config["params"]["Task"]["placement"]["placement_object"]]
        placement_pose = self.placement_object.data.root_link_state_w
        placement_xyz = placement_pose[:, :3]
        placment_quat = last_obs["policy"]["ee_pose"][:, 3:7]
        placement_xyz[:, 2] += (
            self.placement_height[0] +
            (self.placement_height[1] - self.placement_height[0]) *
            torch.rand(1)).to(self.device)
        if target_pos is not None:
            placement_xyz[:, :3] = target_pos
            placement_xyz[:, 2] += 0.11

        ee_pose, traj = self.planner.plan_motion(
            last_obs["policy"]["joint_pos"], placement_xyz, placment_quat)
        if ee_pose is None:
            return None

        curobo_target_positions = ee_pose.ee_position
        curobo_targe_quaternion = ee_pose.ee_quaternion

        curobo_target_ee_pos = torch.cat([
            curobo_target_positions, curobo_targe_quaternion,
            torch.zeros(len(curobo_targe_quaternion), 1).to(self.device)
        ],
                                         dim=1)

        _, placement_ee_traj = curobo2robot_actions(curobo_target_ee_pos,
                                                    self.device)
        placement_ee_traj[:, -1] = -1
        open_gripper_traj = placement_ee_traj[-1].clone().unsqueeze(
            0).repeat_interleave(self.open_gripper_horizon, dim=0)
        open_gripper_traj[:, -1] = 1
        placement_ee_traj = torch.cat([placement_ee_traj, open_gripper_traj],
                                      dim=0)
        return placement_ee_traj

    def step_planner(self, last_obs, skip_frame=2):

        if self.grasp_env.target_ee_traj is None:
            return
        traj = self.grasp_env.target_ee_traj.clone()
        traj = traj[::skip_frame]
        rewards_buffer = []

        for action in traj:
            if self.use_relative_pose:
                actions = process_action(action.unsqueeze(0),
                                         use_relative_pose=True,
                                         robot=self.env.scene["robot"],
                                         device=self.device)
            else:
                actions = action.unsqueeze(0)
            next_obs, rewards, terminated, time_out, info = self.env.step(
                actions)
            success_or_not, _ = criterion_pick_place(
                self.env,
                self.target_object_name,
                self.placement_object_name,
                self.bbox_region,
                args_cli=self.args_cli)

            last_obs["policy"]["success_or_not"] = success_or_not.unsqueeze(1)
            update_buffer(self, next_obs, last_obs, actions, rewards,
                          terminated, time_out)

            last_obs = copy.deepcopy(next_obs)
            rewards_buffer.append(rewards.cpu().numpy())

        grasp_success = self.grasp_env.success_or_not(next_obs)

        success_or_not, _ = criterion_pick_place(self.env,
                                                 self.target_object_name,
                                                 self.placement_object_name,
                                                 self.bbox_region,
                                                 args_cli=self.args_cli)
        if self.args_cli.approach_only:
            if success_or_not:
                self.collector_interface.add_demonstraions_to_buffer(
                    self.obs_buffer, self.action_buffer, self.rewards_buffer,
                    self.does_buffer, self.next_obs_buffer)
        grasp_success = True
        if grasp_success:
            if self.args_cli.pick_only:
                self.collector_interface.add_demonstraions_to_buffer(
                    self.obs_buffer, self.action_buffer, self.rewards_buffer,
                    self.does_buffer, self.next_obs_buffer)

            else:

                placement_ee_traj = self.sample_placement_pose(
                    next_obs,
                    target_pos=next_obs["policy"]["target_object_pos"])
                if placement_ee_traj is not None:
                    placement_ee_traj = placement_ee_traj[::skip_frame]
                    for action in placement_ee_traj:

                        if self.use_relative_pose:
                            actions = process_action(
                                action.unsqueeze(0),
                                use_relative_pose=True,
                                robot=self.env.scene["robot"],
                                device=self.device)
                        else:
                            actions = action.unsqueeze(0)
                        next_obs, rewards, terminated, time_out, info = self.env.step(
                            actions)
                        success_or_not, _ = criterion_pick_place(
                            self.env,
                            self.target_object_name,
                            self.placement_object_name,
                            self.bbox_region,
                            args_cli=self.args_cli)

                        last_obs["policy"][
                            "success_or_not"] = success_or_not.unsqueeze(1)
                        update_buffer(self, next_obs, last_obs, actions,
                                      rewards, terminated, time_out)
                        last_obs = copy.deepcopy(next_obs)
                        rewards_buffer.append(rewards.cpu().numpy())
                    success, _ = criterion_pick_place(
                        self.env,
                        self.target_object_name,
                        self.placement_object_name,
                        self.bbox_region,
                        args_cli=self.args_cli)

                    if success and self.args_cli.save_path is not None:
                        self.collector_interface.add_demonstraions_to_buffer(
                            self.obs_buffer, self.action_buffer,
                            self.rewards_buffer, self.does_buffer,
                            self.next_obs_buffer)
                        rewards_buffer = np.concatenate(rewards_buffer)
                        plt.plot(np.arange(len(rewards_buffer)),
                                 rewards_buffer)
                        plt.show()

        reset_buffer(self)
