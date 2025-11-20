import torch
import isaaclab.utils.math as math_utils
from isaaclab.sensors.camera.utils import obtain_target_quat_from_multi_angles
from scripts.workflows.hand_manipulation.utils.cloudxr.action_manager_wrapper import ActionManagerWrapper

import random
from isaaclab.managers import SceneEntityCfg


class ArmHandWrapper(ActionManagerWrapper):

    def __init__(self, env, teleop_interface, args_cli, env_cfg, teleop_config,
                 visionpro_client):
        self.teleop_interface = teleop_interface
        self.env = env
        self.args_cli = args_cli
        self.env_cfg = env_cfg
        self.add_left_hand = args_cli.add_left_hand
        self.add_right_hand = args_cli.add_right_hand
        self.device = env.device
        super().__init__(env, args_cli, env_cfg, teleop_config,
                         visionpro_client)

        self.teleop_interface._retargeters[
            0].left_hand_pos_offset = self.left_hand_pos_offset.cpu().numpy(
            )[0]
        self.teleop_interface._retargeters[
            0].right_hand_pos_offset = self.right_hand_pos_offset.cpu().numpy(
            )[0]

        self.task = ("place" if "Place" in self.args_cli.task else
                     "open" if "Open" in self.args_cli.task else "grasp")

        self.init_setting()

        self.reset_teleoperation()

    def init_setting(self):

        for i in range(30):
            self.env.step(self.init_actions)

        self.init_rigid_object()

    def init_rigid_object(self):

        self.env_ids = torch.arange(self.env.num_envs, device=self.device)

        if self.task == "grasp":

            self.rigid_object_list = list(self.env_cfg["params"].get(
                "RigidObject", {}).keys())
            self.rigid_object_setting = self.env_cfg["params"].get(
                "RigidObject", {})
            self.num_rigid_objects = len(self.rigid_object_list)
        elif self.task == "open":

            from isaaclab_tasks.manager_based.manipulation.inhand.utils.open.config_cluster_rigids import define_articulation_objects, define_rigid_objects
            articulation_object_setting = self.env_cfg["params"].get(
                "ArticulationObject")
            object_config = self.env_cfg["params"]["RigidObject"]

            self.articulation_objects = list(
                articulation_object_setting.keys())
            num_articulation_object = len(self.articulation_objects)
            self.rigid_object_list = list(
                self.env_cfg["params"]["RigidObject"].keys())

            num_pick_object = len(self.rigid_object_list)
            define_rigid_objects(self,
                                 env_ids=self.env_ids,
                                 num_object=num_pick_object,
                                 object_config=object_config,
                                 objects_list=self.rigid_object_list,
                                 name="pick")

            define_articulation_objects(
                self,
                env_ids=self.env_ids,
                num_object=num_articulation_object,
                articulation_object_setting=articulation_object_setting,
            )
            self.pose_pick_range = self.env_cfg["params"][
                "default_root_state"]["pose_range"]

        elif self.task == "place":

            self.rigid_object_list = self.env_cfg["params"].get(
                "pick_object_names", None)

            self.rigid_object_setting = self.env_cfg["params"].get(
                "RigidObject", {})
            self.place_object_list = self.env_cfg["params"].get(
                "place_object_names", None)

        self.pose_range = self.env_cfg["params"]["default_root_state"][
            "pose_range"]

        self.root_state = torch.as_tensor(
            self.env_cfg["params"]["default_root_state"]["pos"]).unsqueeze(
                0).to(self.device).repeat_interleave(self.env.num_envs, dim=0)

        root_quat = obtain_target_quat_from_multi_angles(
            self.env_cfg["params"]["default_root_state"]["rot"]["axis"],
            self.env_cfg["params"]["default_root_state"]["rot"]["angles"])

        root_quat = torch.as_tensor(root_quat).unsqueeze(0).to(
            self.device).repeat_interleave(self.env.num_envs, dim=0)
        self.root_state = torch.cat([self.root_state, root_quat],
                                    dim=1).to(self.device)
        self.object_count = 0

        self.num_frame = 0

    def reset_root_state_uniform(self, env,
                                 pose_range: dict[str, tuple[float, float]],
                                 root_states, asset_name):

        # extract the used quantities (to enable type-hinting)
        asset = env.scene[asset_name]

        root_velocity = asset.data.default_root_state[self.env_ids].clone()

        # poses
        range_list = [
            pose_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]

        ranges = torch.tensor(range_list, device=asset.device)
        rand_samples = math_utils.sample_uniform(ranges[:, 0],
                                                 ranges[:, 1],
                                                 (len(self.env_ids), 6),
                                                 device=asset.device)

        positions = root_states[:, 0:3] + env.scene.env_origins[
            self.env_ids] + rand_samples[:, 0:3]

        orientations_delta = math_utils.quat_from_euler_xyz(
            rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        orientations = math_utils.quat_mul(orientations_delta,
                                           root_states[:, 3:7])

        # velocities
        range_list = [(0.0, 0.0)
                      for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=asset.device)
        rand_samples = math_utils.sample_uniform(ranges[:, 0],
                                                 ranges[:, 1],
                                                 (len(self.env_ids), 6),
                                                 device=asset.device)

        velocities = root_velocity[:, 7:13] + rand_samples

        asset.write_root_link_pose_to_sim(torch.cat([positions, orientations],
                                                    dim=-1),
                                          env_ids=self.env_ids)
        asset.write_root_velocity_to_sim(velocities, env_ids=self.env_ids)

        return positions, orientations

    def reset_teleoperation(self, change_obejct=False):

        if change_obejct:
            self.object_count += 1

        self.env.reset()

        if self.task == "grasp":

            self.reset_root_state_uniform(
                self.env, self.pose_range, self.root_state.clone(),
                self.rigid_object_list[self.object_count %
                                       self.num_rigid_objects])
        elif self.task == "open":
            target_articulation_name = random.choice(self.articulation_objects)
            pick_object_name = self.rigid_object_list[self.object_count % len(
                self.rigid_object_list)]

            from isaaclab_tasks.manager_based.manipulation.inhand.utils.open.config_cluster_rigids import randomize_object_pose
            self.reset_pick_height[:] = 0.20
            randomize_object_pose(self.env,
                                  self.env_ids,
                                  asset_cfgs=[
                                      SceneEntityCfg(pick_object_name),
                                      SceneEntityCfg(target_articulation_name),
                                  ],
                                  min_separation=0.45,
                                  reset_height=[
                                      self.reset_pick_height,
                                      self.reset_articulation_height
                                  ],
                                  max_sample_tries=5000,
                                  range_list=[
                                      self.pose_pick_range,
                                      self.articulation_pose_range
                                  ])
        elif self.task == "place":
            from isaaclab_tasks.manager_based.manipulation.inhand.utils.place.config_cluster_rigids import randomize_object_pose

            pick_object_name = self.rigid_object_list[self.object_count % len(
                self.rigid_object_list)]
            place_object_name = random.choice(self.place_object_list)

            self.rigid_object_setting[pick_object_name]

            reset_height = [
                torch.as_tensor(self.env_cfg["params"]["RigidObject"]
                                [pick_object_name]["pos"][2]).unsqueeze(0) *
                0.0 + 0.15,
                torch.as_tensor(self.env_cfg["params"]["RigidObject"]
                                [place_object_name]["pos"][2]).unsqueeze(0) *
                0.0
            ]

            randomize_object_pose(env=self.env,
                                  env_ids=self.env_ids,
                                  asset_cfgs=[
                                      SceneEntityCfg(pick_object_name),
                                      SceneEntityCfg(place_object_name)
                                  ],
                                  min_separation=0.30,
                                  pose_range=self.pose_range,
                                  reset_height=reset_height,
                                  max_sample_tries=100)

        for i in range(10):
            self.env.step(self.init_actions)
        self.reset()

    def replay_teleoperation_data(self, obs_buffer, actions_buffer):

        if len(obs_buffer) < 10:
            return
        init_setting = obs_buffer[0]["policy"]
        self.env.reset()

        if self.add_left_hand:
            self.env.scene["left_hand"].root_physx_view.set_dof_positions(
                init_setting["left_hand_joint_pos"].clone(),
                indices=torch.arange(self.env.num_envs).to(self.device))

        if self.add_right_hand:
            self.env.scene["right_hand"].root_physx_view.set_dof_positions(
                init_setting["right_hand_joint_pos"].clone(),
                indices=torch.arange(self.env.num_envs).to(self.device))
        for rigid_object in self.rigid_object_list:
            self.env.scene[rigid_object].write_root_pose_to_sim(
                init_setting[f"{rigid_object}_pose"].clone(),
                env_ids=self.env_ids)

        for i in range(10):
            self.env.step(actions_buffer[0])

        for index in range(len(actions_buffer)):
            self.env.step(actions_buffer[index])

    def step_teleoperation(self,
                           teleoperation_active,
                           reset_stopped_teleoperation,
                           change_obejct=False):

        teleop_data, raw_data = self.teleop_interface.advance()

        reset = False

        if reset_stopped_teleoperation:  # reset come first

            self.reset_teleoperation()

            reset = True

            print("Resetting environment...", self.num_frame)
            self.num_frame = 0

        elif not teleoperation_active and change_obejct:
            self.num_frame = 0
            self.reset_teleoperation(change_obejct=True)
            reset = True

        elif teleoperation_active:

            actions = []

            if self.add_left_hand:
                left_hand_pose = self.step_action(
                    teleop_data[0][0],
                    torch.as_tensor(
                        teleop_data[0][-1][6:self.num_hand_joint + 6]).to(
                            self.device).unsqueeze(0), "left",
                    raw_data["left_hand_joint"])

                actions.append(left_hand_pose)

            if self.add_right_hand:

                right_hand_pose = self.step_action(
                    teleop_data[0][1],
                    torch.as_tensor(
                        teleop_data[0][-1][-self.num_hand_joint:]).to(
                            self.device).unsqueeze(0), "right",
                    raw_data["right_hand_joint"])

                actions.append(right_hand_pose)
            actions = torch.cat(actions, dim=1)

            obs, rewards, terminated, time_outs, extras = self.env.step(
                actions)

            done = terminated | time_outs
            obs["policy"]["raw_teleop_data"] = raw_data

            obs["policy"]["dexretargeting_human_data"] = teleop_data[0][-2]
            if not self.ready_for_teleop:
                obs = None
            return obs, rewards, done, extras, actions, False, False

        else:  # no signal from teleoperation
            self.env.sim.render()

        self.num_frame += 1

        return None, None, None, None, None, reset, change_obejct
