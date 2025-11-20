from scripts.workflows.vlm_failure.stack_block.utils.data_wrapper import Datawrapper

from scripts.workflows.automatic_articulation.task.failure_env_grasp import FailureEnvGrasp
import torch
import numpy as np
import copy


class FailureDatawrapper:

    def __init__(self,
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
                 use_demo_data=True,
                 faiulure_type=None,
                 failure_attempt=1):
        # failure type is the type of failure to be simulated: [x_offset,y_offset,z_offset,roll_offset,pitch_offset,yaw_offset,slip,miss_grasp,wrong_sequeeze,]
        self.multi_env = Datawrapper(env,
                                     collision_checker=collision_checker,
                                     use_relative_pose=use_relative_pose,
                                     collect_data=collect_data,
                                     args_cli=args_cli,
                                     env_config=env_config,
                                     filter_keys=filter_keys,
                                     use_joint_pos=use_joint_pos,
                                     load_path=load_path,
                                     use_fps=use_fps,
                                     use_demo_data=use_demo_data,
                                     save_path=save_path)
        self.collect_data = collect_data
        self.failure_attempt = failure_attempt
        self.failure_count = 0

        self.failure_env = FailureEnvGrasp(env=self.multi_env,
                                           failure_type=faiulure_type)
        self.failure_type = faiulure_type
        self.total_frame = self.failure_env.total_frame

        self.assign_failure()
        self.demo_index = 0
        self.multi_env.reset_data_buffer()

    def assign_failure(self):

        if self.failure_type == "slip":
            self.extract_failure_action = self.failure_env.slip_failure_env
        if "offset" in self.failure_type:
            self.extract_failure_action = self.failure_env.xyz_offset_failure_env
        if self.failure_type == "miss_placement":
            self.extract_failure_action = self.failure_env.miss_placement_failure_env
        if self.failure_type == "miss_grasp":
            self.extract_failure_action = self.failure_env.miss_grasp_failure_env
        if self.failure_type == "miss_grasp":
            self.extract_failure_action = self.failure_env.miss_grasp_failure_env

        if self.failure_type in [
                "mistake_pickup", "mistake_place", "mistake_pickup_place"
        ]:
            self.extract_failure_action = self.failure_env.mistake_pickup_place_failure_env

    def reset_demo_env(self):
        self.demo = self.multi_env.extract_data(
            self.multi_env.collector_interface.raw_data, self.demo_index)
        last_obs, _ = self.multi_env.env.reset()

        obs = self.demo["obs"]

        # reset robot and object poses
        self.multi_env.robot.root_physx_view.set_dof_positions(
            torch.as_tensor(obs["joint_pos"][0]).unsqueeze(0).to(
                self.multi_env.device), self.multi_env.env_ids)

        # reset obqect poses
        keys = list(self.multi_env.env.scene.rigid_objects.keys())
        for key in keys:

            self.multi_env.env.scene.rigid_objects[key].write_root_pose_to_sim(
                torch.as_tensor(obs[key][0][..., :7]).unsqueeze(0).to(
                    self.multi_env.device), self.multi_env.env_ids)

        # extract the token and the target object
        self.multi_env.token = obs["language_intruction"][0].decode("utf-8")
        parts = self.multi_env.token.split("Your task is to pick up ")[1]
        target_object_name, placement_object_name = parts.split(
            " and place it onto ")
        self.multi_env.target_object_name, self.multi_env.placement_object_name = target_object_name.lstrip(
            "the "), placement_object_name.strip("the ").strip(".")
        self.multi_env.assign_objects(self.multi_env.target_object_name,
                                      self.multi_env.placement_object_name)

        for i in range(20):

            observation, reward, terminate, time_out, info = self.multi_env.env.step(
                torch.as_tensor(self.demo["actions"][0]).unsqueeze(0).to(
                    self.multi_env.device), )
            last_obs = observation
        return last_obs

    def step_failure_env(self, last_obs, skip_frame=2):
        action = copy.deepcopy(np.asarray(self.demo["actions"]))
        # rollout success data
        if self.failure_count == 0:
            last_obs = self.multi_env.step_env(
                torch.as_tensor(action).to(self.multi_env.device), last_obs)
            if not self.multi_env.env_placement.success_or_not(last_obs):
                self.multi_env.reset_data_buffer()
                self.demo_index += 1
                return None

        success_buffer = [
            copy.deepcopy(self.multi_env.obs_buffer),
            copy.deepcopy(self.multi_env.actions_buffer),
            copy.deepcopy(self.multi_env.rewards_buffer),
            copy.deepcopy(self.multi_env.does_buffer)
        ]

        last_obs = self.reset_demo_env()

        failure_action, failure_reasoning = self.extract_failure_action(
            demo=self.demo, last_obs=last_obs)
        if failure_action is None:
            return False

        failure_action = failure_action[::skip_frame]

        last_obs = self.multi_env.step_env(failure_action, last_obs)

        if not self.multi_env.env_placement.success_or_not(
                last_obs) or self.failure_type in [
                    "miss_placement", "mistake_pickup", "mistake_place",
                    "mistake_pickup_place"
                ]:
            # add failure reasoning
            for index in range(len(self.multi_env.obs_buffer)):
                self.multi_env.obs_buffer[index]["failure_reasoning"] = [
                    failure_reasoning
                ]
            self.multi_env.cache_data()
            self.demo_index += 1
            self.failure_count = 0
        else:
            self.failure_count += 1
            if self.failure_count >= self.failure_attempt:
                self.failure_count = 0
                self.demo_index += 1

        self.multi_env.reset_data_buffer(
            success_buffer=success_buffer if self.failure_count > 0 else None)

        return None

    def step_failure_env_espisode(self, last_obs, skip_frame=2):

        action = copy.deepcopy(np.asarray(self.demo["actions"]))

        last_obs = self.multi_env.step_env(
            torch.as_tensor(action).to(
                self.multi_env.device)[self.total_frame::skip_frame], last_obs)
        if not self.multi_env.env_placement.success_or_not(
                last_obs) or self.failure_type in [
                    "miss_placement", "mistake_pickup", "mistake_place",
                    "mistake_pickup_place"
                ]:
            self.multi_env.cache_data()
        self.multi_env.reset_data_buffer()
        self.demo_index += 1
