from scripts.workflows.utils.multi_datawrapper import MultiDatawrapper
import isaaclab.utils.math as math_utils
import torch
from scipy.spatial.transform import Rotation as R
from scripts.workflows.hand_manipulation.env.teleop_env.retarget_arm import RetargetArm


class ReplayDexretargetDataset:

    def __init__(self, args_cli, env_config, env):
        self.args_cli = args_cli
        self.env_config = env_config
        self.env = env
        self.use_joint_pose = True if "Player" in args_cli.task else False
        self.collector_interface = MultiDatawrapper(
            args_cli,
            env_config,
            load_path=args_cli.load_path,
            save_path=args_cli.save_path,
        )
        self.raw_data = self.collector_interface.raw_data["data"]

        self.num_trajectories = len(self.raw_data)

        if self.args_cli.add_left_hand:
            self.hand_side = "left"
        elif self.args_cli.add_right_hand:
            self.hand_side = "right"
        self.init_mean = torch.as_tensor(
            [[0.55, 0.1, 0.0, 0.707, 0.0, 0.0, 0.707]]).to(self.env.device)
        self.device = env.device

        if self.env_config["params"]["arm_type"] is not None:

            self.retarget_arm = RetargetArm(
                env,
                args_cli,
                env_config,
                collector_interface=self.collector_interface)

            self.run = self.run_with_arm
        else:

            self.run = self.run_without_arm

    def prepocess(self, index):

        demo_data = self.raw_data[f"demo_{int(index)}"]
        obs_buffer = demo_data["obs"]
        actions_buffer = demo_data["actions"]
        realign_obs_buffer = self.realign_scene(obs_buffer, index)

        return realign_obs_buffer, actions_buffer

    def run_with_arm(self, index):
        realign_obs_buffer, actions_buffer = self.prepocess(index)
        if realign_obs_buffer is None:
            return None

        success = self.retarget_arm.run_with_arm(realign_obs_buffer,
                                                 actions_buffer)
        return success

    def get_transformation_matrix(self, ee_pose, index):
        axis_angle = math_utils.axis_angle_from_quat(ee_pose[-1, 3:7])

        if self.args_cli.data_type == "dexycb":

            if self.hand_side == "left":
                if abs(axis_angle[2]) > 3.14 / 2:
                    self.transformation_matrix = math_utils.pose_to_transformations(
                        torch.tensor([[0.0, 0.0, 0.0, 0.707, 0, 0,
                                       0.707]]).to(self.device))

                else:
                    self.transformation_matrix = math_utils.pose_to_transformations(
                        torch.tensor([[0.0, 0.0, 0.0, 1, 0, 0,
                                       0]]).to(self.device))
            elif self.hand_side == "right":
                if abs(axis_angle[2]) < 3.14 / 2:
                    self.transformation_matrix = math_utils.pose_to_transformations(
                        torch.tensor([[0.0, 0.0, 0.0, 0.707, 0, 0,
                                       0.707]]).to(self.device))
                else:
                    self.transformation_matrix = math_utils.pose_to_transformations(
                        torch.tensor([[0.0, 0.0, 0.0, 1, 0, 0,
                                       0]]).to(self.device))
        elif self.args_cli.data_type == "hocap":
            if self.hand_side == "right":
                if abs(axis_angle[2]) > 3.14 / 2:
                    self.transformation_matrix = math_utils.pose_to_transformations(
                        torch.tensor([[0.0, 0.0, 0.0, 1.0, 0, 0,
                                       0.0]]).to(self.device))
                else:
                    self.transformation_matrix = math_utils.pose_to_transformations(
                        torch.tensor([[0.0, 0.0, 0.0, 1, 0, 0,
                                       0]]).to(self.device))

    def realign_scene(self, obs_buffer, index):

        ee_pose = torch.tensor(obs_buffer["ee_pose"]).to(self.device)

        realign_obs_buffer = {}

        self.get_transformation_matrix(ee_pose, index)
        delta_height = None
        delta_x, delta_y = 0.0, 0.0
        for name in obs_buffer.keys():
            pose_state = torch.tensor(obs_buffer[name]).to(self.device)
            pose_state_matrix = math_utils.pose_to_transformations(pose_state)

            transformed_pose = torch.bmm(
                self.transformation_matrix.repeat_interleave(
                    len(pose_state_matrix), 0), pose_state_matrix)
            transformed_pose = math_utils.pose_from_transformations(
                transformed_pose).to(self.device)

            if name == "ee_pose":

                realign_obs_buffer[name] = transformed_pose
            else:
                delta_height = max(transformed_pose[:, 2]) - min(
                    transformed_pose[:, 2])
                if delta_height < 0.1:
                    continue
                else:
                    delta_x = self.init_mean[:, 0] - transformed_pose[0, 0]
                    delta_y = self.init_mean[:, 1] - transformed_pose[0, 1]
                    realign_obs_buffer[name] = transformed_pose
        if delta_height is None:
            return None
        for name in realign_obs_buffer.keys():
            realign_obs_buffer[name][:, 0] += delta_x
            realign_obs_buffer[name][:, 1] += delta_y

        if self.args_cli.target_object is not None:
            if self.args_cli.target_object not in realign_obs_buffer.keys():
                return None

        return realign_obs_buffer

    def run_without_arm(self, index):

        realign_obs_buffer, actions_buffer = self.prepocess(index)
        joint_limits = self.env.scene[
            f"{self.hand_side}_hand"]._data.joint_limits

        for i in range(len(actions_buffer)):

            # load finger actions

            actions = torch.tensor(actions_buffer[i][..., -16:],
                                   device=self.device).unsqueeze(0)

            actions = torch.clip(actions, joint_limits[..., 0],
                                 joint_limits[..., 1])

            # control the hand pose
            raw_root_state = self.env.scene[
                f"{self.hand_side}_hand"]._data.root_state_w
            raw_root_state[:, :3] = torch.tensor(
                realign_obs_buffer["ee_pose"][i][:3]).to(self.device)

            raw_root_state[:, 3:7] = torch.tensor(
                realign_obs_buffer["ee_pose"][i][3:7]).to(self.device)
            # raw_root_state[:, 3:7] = torch.as_tensor(
            #     [-0.7153, 0.0027, 0.0026, -0.6988]).to(self.device)
            self.env.scene[f"{self.hand_side}_hand"].write_root_pose_to_sim(
                raw_root_state[:, :7],
                torch.arange(self.env.num_envs).to(self.device))

            # # control the finger
            # if i < 25:
            #     actions *= 0.0

            # load ycb object pose

            self.update_manipulated_pose(realign_obs_buffer, i)
            obs, rewards, terminated, time_outs, extras = self.env.step(
                actions)
            # print(obs["policy"]["left_contact_obs"])

        realign_obs_buffer.pop("ee_pose")
        object_name = list(realign_obs_buffer.keys())[0]
        if self.env.scene[object_name]._data.root_state_w[:, 2] > 0.10:
            success = True
        else:
            success = False

        self.env.reset()
        return success

    def update_manipulated_pose(self, realign_obs_buffer, i):

        for ycb_name in realign_obs_buffer.keys():
            if ycb_name not in self.env.scene.rigid_objects.keys():
                continue

            ycb_pose = torch.tensor(realign_obs_buffer[ycb_name][i]).to(
                self.device)

            ycb_pose = ycb_pose.unsqueeze(0)

            self.env.scene.rigid_objects[ycb_name].write_root_pose_to_sim(
                ycb_pose.unsqueeze(0),
                torch.arange(self.env.num_envs).to(self.device))
