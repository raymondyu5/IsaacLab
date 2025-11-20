from tools.curobo_planner import IKPlanner, MotionPlanner
from source.isaaclab_tasks.isaaclab_tasks.manager_based.manipulation.cabinet import mdp
import torch
import isaaclab.utils.math as math_utils
from scripts.workflows.automatic_articulation.utils.process_action import process_action, get_robottip_pose
from scripts.workflows.utils.multi_datawrapper import MultiDatawrapper
from scripts.workflows.open_policy.utils.buffer_utils import reset_buffer, update_buffer
import numpy as np


class ReplayMotionPlanReach:

    def __init__(self, env, reach_config, args_cli, use_ik_pose=False):
        self.env = env
        self.robot = env.scene["robot"]
        self.device = self.env.device

        self.reach_config = reach_config
        self.use_ik_pose = use_ik_pose

        self.collector_interface = MultiDatawrapper(
            args_cli,
            reach_config,
            filter_keys=[],
            load_path=args_cli.load_replay_path,
            save_path=args_cli.save_replay_path,
            use_fps=False,
            use_joint_pos=False if "joint" not in args_cli.task else True,
            normalize_action=False)
        self.args_cli = args_cli
        if self.args_cli.save_replay_path is not None:           
            self.collector_interface.init_collector_interface()
            reset_buffer(self)
        self.init_planner()

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
        self.ik_planner = self.planner.curobo_ik
        self.env.scene["robot"]._data.joint_pos

        self.init_ee_pose = torch.as_tensor(
            self.reach_config["params"]["Task"]["init_ee_pose"]).unsqueeze(
                0).to(self.device).repeat_interleave(self.env.num_envs, 0)
        # result = self.planner.plan_motion(self.env.scene["robot"]._data.joint_pos,self.init_ee_pose[:, :3], self.init_ee_pose[:, 3:7])

        self.position_range = torch.as_tensor(
            self.reach_config["params"]["Task"]["position_range"]).to(
                self.device)
        self.init_jpos = self.ik_planner.plan_motion(self.init_ee_pose[:, :3],
                                                     self.init_ee_pose[:, 3:7])
        self.env_indices = torch.arange(self.env.num_envs, device=self.device)
        self.demo_index = 0

    def add_noise_to_position(self, ee_pose):

        ee_pose[:, :3] = math_utils.sample_uniform(self.position_range[1][:3],
                                                   self.position_range[0][:3],
                                                   ee_pose[:, :3].shape,
                                                   ee_pose.device)
        delta_euler_angle_noise = math_utils.sample_uniform(
            self.position_range[1][3:6], self.position_range[0][3:6],
            ee_pose[:, :3].shape, ee_pose.device)
        delta_quat_noise = math_utils.quat_from_euler_xyz(
            delta_euler_angle_noise[:, 0], delta_euler_angle_noise[:, 1],
            delta_euler_angle_noise[:, 2])
        ee_pose[:, 3:7] = math_utils.quat_mul(delta_quat_noise, ee_pose[:,
                                                                        3:7])

        return ee_pose

    def reset_env(self):

        self.cur_obs = self.collector_interface.raw_data["data"][
            f"demo_{self.demo_index}"]["obs"]
        self.cur_actions = torch.as_tensor(
            np.array(self.collector_interface.raw_data["data"]
                     [f"demo_{self.demo_index}"]["actions"])).to(
                         self.device)
        self.env.reset()
        reset_joint_pos = torch.as_tensor(
            self.cur_obs["joint_pos"][0]).unsqueeze(0).to(self.device)
        
        need_dof = self.env.scene["robot"].data.reset_joint_pos
        need_dof[:,:7] = reset_joint_pos[:,:7]
        self.env.scene["robot"].root_physx_view.set_dof_positions(
            need_dof, self.env_indices)

        if not self.use_ik_pose:
            abs_poses = torch.cat([
                reset_ee_position, reset_ee_quat,
                torch.zeros(1, 1).to(self.device)
            ],
                                  dim=1)
            for i in range(20):
                obs, rewards, terminated, time_outs, extras = self.env.step(
                    abs_poses)
        else:

            for i in range(20):
                obs, rewards, terminated, time_outs, extras = self.env.step(
                    torch.zeros((self.env.num_envs, 7)).to(self.device))
        self.demo_index += 1
        return obs

    def step_motion(self, last_obs):
       

        for index, action in enumerate(self.cur_actions):

            next_obs, rewards, terminated, time_outs, extras = self.env.step(
                action.unsqueeze(0))
            if self.args_cli.save_replay_path is not None: 
                update_buffer(self, next_obs, last_obs, action.unsqueeze(0),
                            rewards, terminated, time_outs)
            last_obs = next_obs
           
        if self.args_cli.save_replay_path is not None:         
            self.collector_interface.add_demonstraions_to_buffer(
                self.obs_buffer, self.action_buffer, self.rewards_buffer,
                self.does_buffer, self.next_obs_buffer)
            reset_buffer(self)

        # ee_pose, _ = get_robottip_pose(self.robot,
        #                                self.device,
        #                                use_gripper_offset=False)
        # print("delta pose",torch.linalg.norm(ee_pose-(init_ee_pose+action.unsqueeze(0)[:,:3])),
        #       torch.linalg.norm(ee_pose[0,:3]-target_poses[-1,:3]),)
