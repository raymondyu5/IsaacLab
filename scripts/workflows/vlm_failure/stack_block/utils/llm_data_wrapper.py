from scripts.workflows.vlm_failure.stack_block.utils.data_wrapper import Datawrapper

import torch
import numpy as np
import copy

from scripts.workflows.vlm_failure.stack_block.utils.llm_openai_agent import LMPAgent
import re

import types
from types import MethodType
from scripts.workflows.vlm_failure.stack_block.utils.correction_task import CorrectGraspEnv


class LLMDatawrapper(CorrectGraspEnv):

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
        use_demo_data=True,
        failure_attempt=5,
    ):
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

        self.LMPAgent = LMPAgent(
            "source/config/llm_prompt/block_stack/prompt_blockstack.txt",
            "source/config/llm_prompt/block_stack/prompt_failure_template.txt")
        super().__init__()
        self.collect_data = collect_data
        self.failure_count = 0
        self.failure_attempt = failure_attempt
        self.previous_failure_reasoning = []
        self.previous_llm_feedback = []
        self.current_failure_reasoning = []

        self.demo_index = 0
        self.multi_env.reset_data_buffer()

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

        self.gt_target_object_name = target_object_name.lstrip("the ")
        self.gt_placement_object_name = placement_object_name.strip(
            "the ").strip(".")
        self.multi_env.assign_objects(self.multi_env.target_object_name,
                                      self.multi_env.placement_object_name)
        # failure_reasoning and language_intruction
        self.failure_reasoning = self.demo["obs"]["failure_reasoning"][
            0].decode("utf-8")
        self.language_instruction = self.demo["obs"]["language_intruction"][
            0].decode("utf-8")
        self.current_failure_reasoning = copy.deepcopy(self.failure_reasoning)

        for i in range(20):

            observation, reward, terminate, time_out, info = self.multi_env.env.step(
                torch.as_tensor(self.demo["actions"][0]).unsqueeze(0).to(
                    self.multi_env.device), )
            last_obs = observation
        return last_obs

    def step_env_espisode(self, last_obs, skip_frame=2):

        action = copy.deepcopy(np.asarray(self.demo["actions"]))

        last_obs = self.multi_env.step_env(
            torch.as_tensor(action).to(self.multi_env.device), last_obs)
        if not self.multi_env.env_placement.success_or_not(last_obs):
            self.multi_env.cache_data()
        self.multi_env.reset_data_buffer()
        self.demo_index += 1

    def correct_env_espiode(self, last_obs, skip_frame=2):
        # action = copy.deepcopy(np.asarray(self.demo["actions"]))

        # start_frame = int(len(action) / 2)
        # last_obs = self.multi_env.step_env(
        #     torch.as_tensor(action).to(
        #         self.multi_env.device)[start_frame::skip_frame], last_obs)

        # last_obs = self.reset_demo_env()
        try:
            llm_code = self.LMPAgent.prompt_llm(
                self.language_instruction, self.failure_reasoning,
                self.previous_failure_reasoning, self.previous_llm_feedback)

            exec(llm_code)
            gen_plan_func = locals()['gen_plan']
            self.gen_plan = MethodType(gen_plan_func, self)

            success_or_not, final_obs = self.gen_plan(last_obs=last_obs)
        except Exception as e:
            print(e)
            success_or_not = False
            self.llm_feedback = "cannot retrieve the plan"

        if success_or_not:
            print('=========================')
            print("replan Success")

            # print("gt pick and place object", self.gt_target_object_name,
            #       self.gt_placement_object_name)
            # print("new pick and place object", new_pickup_object_name,
            #       new_placement_object_name)

            self.llm_feedback = []
            self.previous_failure_reasoning = []
            self.previous_llm_feedback = []
            self.failure_count = 0
            self.demo_index += 1
        else:
            print(f"{self.llm_feedback} ,for {self.failure_count} times")
            self.previous_failure_reasoning = []
            self.failure_count += 1
            self.previous_llm_feedback = []

            if self.failure_count >= self.failure_attempt:
                self.demo_index += 1
                self.failure_count = 0
        # self.multi_env.reset_data_buffer()
