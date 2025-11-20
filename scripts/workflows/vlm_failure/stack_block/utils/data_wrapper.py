from scripts.workflows.utils.multi_datawrapper import MultiDatawrapper
from scripts.workflows.vlm_failure.stack_block.task.env_grasp import GrasperEnv
from scripts.workflows.vlm_failure.stack_block.task.env_placement import PlacementEnv
from tools.curobo_planner import MotionPlanner
from source.isaaclab_tasks.isaaclab_tasks.manager_based.manipulation.cabinet import mdp
import torch
import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
import numpy as np
from source.isaaclab_tasks.isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events
from scripts.workflows.automatic_articulation.utils.process_action import process_action


class Datawrapper:

    def __init__(
        self,
        env,
        collision_checker,
        use_relative_pose,
        collect_data=None,
        args_cli=None,
        env_config=None,
        filter_keys=None,
        load_path=None,
        save_path=None,
        use_fps=False,
        use_joint_pos=False,
        use_demo_data=False,
    ):
        self.env = env
        self.reset_all = True
        self.env_config = env_config
        self.use_demo_data = use_demo_data
        self.args_cli = args_cli
        self.use_joint_pos = use_joint_pos
        self.use_relative_pose = use_relative_pose
        self.device = self.env.device
        self.env_ids = torch.arange(self.env.num_envs).to(self.device)
        self.collision_checker = collision_checker

        if collect_data:
            # init data saving and loading
            self.collector_interface = MultiDatawrapper(
                args_cli,
                env_config,
                filter_keys,
                load_path=load_path,
                save_path=save_path,
                use_fps=use_fps,
                use_joint_pos=use_joint_pos)
            self.collector_interface.init_collector_interface()
            self.reset_data_buffer()
        self.init_planner()
        self.robot = self.env.scene["robot"]

        # init grasp env
        self.env_grasp = GrasperEnv(
            env,
            self.planner,
            use_relative_pose=self.use_relative_pose,
            collision_checker=collision_checker,
            env_config=self.env_config,
        )
        self.env_placement = PlacementEnv(
            env,
            self.planner,
            use_relative_pose=self.use_relative_pose,
            collision_checker=collision_checker,
            env_config=self.env_config)
        self.init_setting()

    def extract_data(self, raw_data, demo_index):

        demo = raw_data["data"][f"demo_{demo_index}"]

        return demo

    def assign_objects(self, target_object_name, placement_object_name):
        # set to the grasp and placement env
        self.env_grasp.grasp_object = self.env.scene[target_object_name]
        self.env_grasp.grasp_object_name = target_object_name

        self.env_placement.target_object = self.env.scene[target_object_name]
        self.env_placement.target_object_name = target_object_name
        self.env_placement.placement_object = self.env.scene[
            placement_object_name]
        if "cube" in target_object_name.lower():
            self.env_grasp.grasp_mode = "block"
        else:
            self.env_grasp.grasp_mode = None

    def init_setting(self):
        self.robot_pose_random_range = self.env_config["params"]["Task"][
            "robot_pose_random_range"]
        self.ee_random_range = self.env_config["params"]["Task"][
            "ee_random_range"]
        self.sample_object_setting = self.env_config["params"]["Task"][
            "sample_object_setting"]
        self.success_pick_threhold = self.env_config["params"]["Task"][
            "success_pick_threhold"]
        self.target_placement_object = self.env_config["params"]["Task"][
            "target_placement_object"]
        self.curobo_planner_length = self.env_config["params"]["Task"][
            "curobo_planner_length"]

    def reset_data_buffer(self, success_buffer=None):
        if success_buffer is None:
            self.obs_buffer = []
            self.actions_buffer = []
            self.rewards_buffer = []
            self.does_buffer = []
        else:

            self.obs_buffer = success_buffer[0]
            self.actions_buffer = success_buffer[1]
            self.rewards_buffer = success_buffer[2]
            self.does_buffer = success_buffer[3]

    def init_planner(self):
        self.planner = MotionPlanner(
            self.env,
            collision_checker=self.collision_checker,
            reference_prim_path="/World/envs/env_0/Robot",
            ignore_substring=[
                "/World/envs/env_0/Robot",
                "/World/GroundPlane",
                "/World/collisions",
                "/World/light",
                "/curobo",
            ],
        )

    def generate_token(self):

        keys = list(self.env.scene.rigid_objects.keys())
        keys_str = ", ".join(keys)
        self.token = f"In the scene, you will find the following objects: {keys_str}. Your task is to pick up the {self.target_object_name} and place it onto the {self.placement_object_name}."

    def sample_object_pose(self, observation):

        keys = list(self.env.scene.rigid_objects.keys())
        asset_cfgs = [SceneEntityCfg(key) for key in keys]

        franka_stack_events.randomize_object_pose(
            self.env,
            self.env_ids,
            asset_cfgs,
            min_separation=self.sample_object_setting[
                "min_separation"],  # Corrected the argument assignment
            pose_range=self.sample_object_setting["pose_range"])

        # remove the target placement object
        if self.target_placement_object is not None:
            keys.remove(self.target_placement_object)

        self.target_object_name = np.random.choice(keys)
        # set the information of the grasp sampler
        self.target_grasp_object = self.env.scene[self.target_object_name]
        self.env_grasp.grasp_object = self.target_grasp_object
        self.env_grasp.grasp_object_name = self.target_object_name

        # remove the object from the scene
        keys.remove(self.target_object_name)

        if self.target_placement_object is not None:
            self.placement_object_name = self.target_placement_object
        else:
            self.placement_object_name = np.random.choice(keys)
        # load the info into placement env
        self.placement_object = self.env.scene[self.placement_object_name]
        self.env_placement.target_object_name = self.target_object_name
        self.env_placement.target_object = self.target_grasp_object
        self.env_placement.placement_object = self.placement_object

        if "cube" in self.target_object_name.lower():
            self.env_grasp.grasp_mode = "block"
        else:
            self.env_grasp.grasp_mode = None

    def reset_env(self):
        observation, _ = self.env.reset()

        # reset robot pose
        self.robot.root_physx_view.set_dof_positions(
            self.env_grasp.init_jpos[:, :9], self.env_ids)
        mdp.reset_rigid_articulation(self.env, self.env_ids, "robot",
                                     self.robot_pose_random_range)
        self.reset_robot_ee_pose = mdp.add_noise_to_position(
            self.env_grasp.init_ee_pose.clone(),
            self.ee_random_range["position_range"])

        # sample object pose
        self.sample_object_pose(observation)

        # step for a few steps
        for i in range(20):
            if self.use_relative_pose:
                observation, reward, terminate, time_out, info = self.env.step(
                    torch.rand(self.env.action_space.shape, device=self.device)
                    * 0.0)
            else:

                observation, reward, terminate, time_out, info = self.env.step(
                    self.reset_robot_ee_pose.repeat_interleave(
                        self.env.num_envs, 0))
        self.env_grasp.reset(observation, resample=True)
        return observation

    def step_env(self, trajectory, last_obs):
        for i in range(len(trajectory)):
            act = process_action(trajectory[i], self.use_relative_pose,
                                 self.robot, self.device)

            observation, reward, terminate, time_out, info = self.env.step(act)

            last_obs["policy"]["language_intruction"] = [self.token]

            self.obs_buffer.append(last_obs["policy"])
            self.actions_buffer.append(act)
            self.rewards_buffer.append(reward)
            self.does_buffer.append(terminate)
            last_obs = observation
        return observation

    def cache_data(self, ):
        stop = self.collector_interface.add_demonstraions_to_buffer(
            self.obs_buffer,  # Observation buffer
            self.actions_buffer,  # Actions buffer
            self.rewards_buffer,  # Rewards buffer
            self.does_buffer  # Termination status buffer
        )
        return stop

    def step_manipulation(self, observation):
        self.generate_token()
        stop = False

        # grasp action

        grasp_action = self.env_grasp.target_ee_traj
        if grasp_action is not None:
            observation = self.step_env(grasp_action, last_obs=observation)
            grasp_success = self.env_grasp.success_or_not(observation)
            if grasp_success:
                # placement action

                self.env_placement.get_target_placement_traj()
                placement_action = self.env_placement.target_ee_traj
                observation = self.step_env(placement_action, observation)
                placement_success = self.env_placement.success_or_not(
                    observation)
                if placement_success:
                    print("Success")
                    stop = self.cache_data()
        self.reset_data_buffer()
        return stop
