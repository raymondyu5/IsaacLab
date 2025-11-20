import torch
import isaaclab.utils.math as math_utils

import copy


class RetargetArm:

    def __init__(self,
                 env,
                 args_cli=None,
                 env_cfg=None,
                 use_planner=True,
                 collector_interface=None):
        self.env = env
        self.device = env.device
        self.args_cli = args_cli
        self.env_cfg = env_cfg
        self.use_joint_pose = True if "Play" in args_cli.task else False

        if self.env_cfg["params"]["arm_type"] == "franka":
            self.ee_link_name = "panda_link7"

        if self.args_cli.add_left_hand:
            self.hand_side = "left"
        elif self.args_cli.add_right_hand:
            self.hand_side = "right"

        self.use_planner = use_planner
        self.collector_interface = collector_interface
        self.init_settings()

        if self.args_cli.save_path is not None:
            self.init_data_buffer()

    def init_data_buffer(self):
        self.obs_buffer = []
        self.actions_buffer = []
        self.does_buffer = []
        self.rewards_buffer = []
        self.postgrasp_pose_ee = torch.zeros((1, 7)).to(self.device)
        self.pregrasp_pose_ee = torch.zeros((1, 7)).to(self.device)

    def step_env(self, actions, objects_name):
        obs, rewards, terminated, time_outs, extras = self.env.step(actions)

        for object_name in objects_name:
            if object_name not in self.env.scene.rigid_objects.keys():
                continue
            object_pose = self.env.scene[object_name]._data.root_state_w[:, :7]
            obs["policy"][object_name] = object_pose

        obs["policy"]["manipulate_object_name"] = [
            self.manipulated_object_name
        ]
        obs["policy"]["postgrasp_pose_ee"] = self.postgrasp_pose_ee
        obs["policy"]["pregrasp_pose_ee"] = self.pregrasp_pose_ee

        rewards = torch.zeros((len(actions), 1), device=self.device)

        if self.args_cli.save_path:
            self.obs_buffer.append(obs)
            self.actions_buffer.append(actions.clone())
            self.rewards_buffer.append(rewards)
            self.does_buffer.append(terminated)
        return obs, rewards, terminated, time_outs, extras

    def init_settings(self):

        self.num_arm_joints = self.env_cfg["params"]["num_arm_joints"]
        self.num_hand_joints = self.env_cfg["params"]["num_hand_joints"]

        self.pre_grasp_penalty = torch.as_tensor(
            self.env_cfg["params"]["pre_grasp_penalty"]).to(self.device)
        self.post_grasp_penalty = torch.as_tensor(
            self.env_cfg["params"]["post_grasp_penalty"]).to(self.device)

        self.num_hand_joints = self.env_cfg["params"]["num_hand_joints"]
        self.init_hand_qpos = torch.as_tensor([0] * self.num_hand_joints).to(
            self.device).unsqueeze(0)

        from scripts.workflows.hand_manipulation.env.teleop_env.motion_planner_env import ArmMotionPlannerEnv
        self.arm_motion_env = ArmMotionPlannerEnv(
            self.env,
            self.args_cli,
            self.env_cfg,
            collision_checker=self.args_cli.collision_checker)
        self.init_robot_pose()

    def init_robot_pose(self):
        if self.env_cfg["params"].get("init_ee_pose", None) is not None:
            self.init_ee_pose = torch.as_tensor(
                self.env_cfg["params"]["init_ee_pose"]).to(
                    self.device).unsqueeze(0)
            self.init_arm_qpos = self.arm_motion_env.ik_plan_motion(
                self.init_ee_pose)

            self.init_ee_pose = torch.cat(
                [self.init_ee_pose, self.init_hand_qpos], dim=1)
            self.init_robot_qpos = torch.cat(
                [self.init_arm_qpos, self.init_hand_qpos], dim=1)
        else:
            reset_joint_pose = torch.as_tensor(
                self.env_cfg["params"]["reset_joint_pose"]).to(
                    self.device).unsqueeze(0)
            self.init_robot_qpos = torch.cat(
                [self.init_arm_qpos, self.init_hand_qpos], dim=1)

    def pregrasp_pose(self, realign_obs_buffer):

        # lift_bound_index = max_height_index

        if self.args_cli.data_type == "dexycb":
            grasp_frame = torch.argmin(realign_obs_buffer["ee_pose"][10:,
                                                                     2]) + 10
        elif self.args_cli.data_type == "hocap":
            realign_obs_buffer_copy = copy.deepcopy(realign_obs_buffer)
            realign_obs_buffer_copy.pop("ee_pose")
            if list(realign_obs_buffer_copy.keys()) == []:
                return None, None
            object_name = list(realign_obs_buffer_copy.keys())[0]

            init_height_index = realign_obs_buffer[object_name][10, 2]
            lift_bound_index = torch.where(
                realign_obs_buffer[object_name][10:, 2] > (init_height_index +
                                                           0.01))
            if len(lift_bound_index[0]) == 0:
                import pdb
                pdb.set_trace()
            else:
                grasp_frame = lift_bound_index[0][0]

        pregrasp_pose = realign_obs_buffer["ee_pose"][grasp_frame].clone()
        if self.use_planner:

            pregrasp_pose[..., :3] += self.pre_grasp_penalty

            plan_ee_pose, prepostgrasp_arm_pose = self.arm_motion_env.plan_motion(
                pregrasp_pose.unsqueeze(0))
            if prepostgrasp_arm_pose is None:
                return None, None
        else:

            pregrasp_arm_pose = torch.as_tensor(realign_obs_buffer["ee_pose"])

        if self.use_joint_pose:
            pregrasp_arm_pose = prepostgrasp_arm_pose.clone()
        else:

            pregrasp_arm_pose = plan_ee_pose.clone()
        self.pregrasp_pose_ee = pregrasp_pose.clone()

        return pregrasp_arm_pose, pregrasp_pose

    def update_manipulated_pose(self, realign_obs_buffer, i):

        for object_name in realign_obs_buffer.keys():
            if object_name not in self.env.scene.rigid_objects.keys():
                continue

            object_pose = torch.tensor(realign_obs_buffer[object_name][i]).to(
                self.device)

            object_pose = object_pose.unsqueeze(0)
            # object_pose[..., :3] += 0.01
            self.env.scene.rigid_objects[object_name].write_root_pose_to_sim(
                object_pose.unsqueeze(0),
                torch.arange(self.env.num_envs).to(self.device))

    def interplate_finger_action(self, finger_pose, num_finger_action):

        finger_speed = finger_pose / num_finger_action
        arange = torch.arange(num_finger_action).to(self.device).unsqueeze(1)
        finger_mat = finger_speed.repeat_interleave(num_finger_action, 0)

        finger_action = finger_mat * arange

        return finger_action

    def extract_grasp_pose(self, actions_buffer, pregrasp_pose):
        if self.args_cli.collision_checker:
            self.arm_motion_env.motion_planner.clear_obstacles()

        grasp_pose = pregrasp_pose.clone()
        # grasp_pose[..., :3] += self.pre_grasp_penalty
        grasp_pose[..., :3] += self.post_grasp_penalty
        self.postgrasp_pose_ee = grasp_pose.clone()

        # get the grasp pose
        plan_ee_pose, postgrasp_arm_qpos = self.arm_motion_env.plan_motion(
            grasp_pose.unsqueeze(0))
        if postgrasp_arm_qpos is None:
            return None
        if self.use_joint_pose:
            postgrasp_arm_pose = postgrasp_arm_qpos.clone()
        else:
            postgrasp_arm_pose = plan_ee_pose.clone()
        post_grasp_pose = torch.as_tensor(
            self.env.action_space.sample()).repeat_interleave(
                postgrasp_arm_pose.shape[0], 0).to(self.device)
        shorten_grasp_pose = postgrasp_arm_pose[::3]

        post_grasp_pose[:len(shorten_grasp_pose
                             ), :-self.num_hand_joints] = shorten_grasp_pose
        post_grasp_pose[len(shorten_grasp_pose):, :-self.
                        num_hand_joints] = shorten_grasp_pose[-1]

        # get the finger action
        num_finger_action = int(len(postgrasp_arm_pose) * 2 / 3)

        finger_pose = torch.as_tensor(
            actions_buffer[0, -self.num_hand_joints:]).unsqueeze(0).to(
                self.device)

        finger_action = self.interplate_finger_action(finger_pose,
                                                      num_finger_action)
        post_grasp_pose[:num_finger_action,
                        -self.num_hand_joints:] = finger_action
        post_grasp_pose[num_finger_action:,
                        -self.num_hand_joints:] = finger_pose
        return post_grasp_pose

    def step_reset(self):
        for i in range(10):

            if self.use_joint_pose:
                obs, rewards, terminated, time_outs, extras = self.env.step(
                    self.init_robot_qpos)
            else:
                obs, rewards, terminated, time_outs, extras = self.env.step(
                    self.init_ee_pose)

    def pregrasp(self, realign_obs_buffer, object_names):
        if self.args_cli.collision_checker:
            self.arm_motion_env.motion_planner.add_obstacle(
                target_collision_checker_name=self.manipulated_object_name)

        ###### geet the pregrasp pose
        pregrasp_arm_pose, pregrasp_pose = self.pregrasp_pose(
            realign_obs_buffer)

        if pregrasp_arm_pose is None:
            return None, None
        self.update_manipulated_pose(realign_obs_buffer, 0)
        for i in range(pregrasp_arm_pose.shape[0]):

            actions = torch.cat(
                [pregrasp_arm_pose[i].unsqueeze(0), self.init_hand_qpos],
                dim=1).to(self.device)

            obs, rewards, terminated, time_outs, extras = self.step_env(
                actions, object_names)

        return pregrasp_pose, pregrasp_arm_pose

    def postgrasp(self, realign_obs_buffer, actions_buffer, pregrasp_pose,
                  object_names):

        ### generate pose grasp pose
        post_grasp_pose = self.extract_grasp_pose(actions_buffer,
                                                  pregrasp_pose)
        if post_grasp_pose is None:
            return None
        self.update_manipulated_pose(realign_obs_buffer, 0)

        for i in range(post_grasp_pose.shape[0]):

            obs, rewards, terminated, time_outs, extras = self.step_env(
                post_grasp_pose[i].unsqueeze(0), object_names)
        return post_grasp_pose

    def lift_object(self, post_grasp_pose, object_names):

        ee_pose = self.env.scene[
            f"{self.hand_side}_{self.ee_link_name}"]._data.root_state_w[:, :7]
        ee_pose[:, :3] = torch.as_tensor([0.55, 0.0, 0.4]).to(self.device)

        plan_ee_pose, list_arm_qpos = self.arm_motion_env.plan_motion(
            ee_pose=ee_pose, apply_offset=False)

        if list_arm_qpos is None:

            return
        if self.use_joint_pose:
            arm_pose = list_arm_qpos.clone()
        else:
            arm_pose = plan_ee_pose.clone()

        for i in range(arm_pose.shape[0]):
            # load finger actions
            actions = post_grasp_pose[-1].unsqueeze(0).to(self.device)
            actions[:, :-self.num_hand_joints] = torch.as_tensor(
                arm_pose[i]).unsqueeze(0).clone()
            obs, rewards, terminated, time_outs, extras = self.step_env(
                actions, object_names)

    def replay_actions(self, realign_obs_buffer, action_buffer):
        self.env.reset()
        self.step_reset()

        for action in action_buffer:

            obs, rewards, terminated, time_outs, extras = self.env.step(action)

    def wrap_up(self, realign_obs_buffer):
        realign_obs_buffer.pop("ee_pose")
        object_name = list(realign_obs_buffer.keys())
        if len(object_name) == 0:
            success = False
        else:
            if self.env.scene[object_name[0]]._data.root_state_w[:, 2] > 0.10:
                success = True
            else:
                success = False

        if self.args_cli.save_path:

            if success or self.args_cli.collect_all:
                self.collector_interface.add_demonstraions_to_buffer(
                    self.obs_buffer,
                    self.actions_buffer,
                    self.rewards_buffer,
                    self.does_buffer,
                )

            # self.replay_actions(realign_obs_buffer, self.actions_buffer)

        self.env.reset()
        return success

    def run_with_arm(self, realign_obs_buffer, actions_buffer):
        self.init_data_buffer()
        self.step_reset()

        object_names = list(self.env.scene.rigid_objects.keys())
        self.manipulated_object_name = list(realign_obs_buffer.keys())
        self.manipulated_object_name.remove("ee_pose")

        # pregrasp the object
        pregrasp_pose, pregrasp_arm_pose = self.pregrasp(
            realign_obs_buffer, object_names)
        if pregrasp_pose is None:
            return None

        # postgrasp the object
        post_grasp_pose = self.postgrasp(realign_obs_buffer, actions_buffer,
                                         pregrasp_pose, object_names)
        if post_grasp_pose is None:
            return None

        self.lift_object(post_grasp_pose, object_names)

        # wrap up the object
        success = self.wrap_up(realign_obs_buffer)

        return success
