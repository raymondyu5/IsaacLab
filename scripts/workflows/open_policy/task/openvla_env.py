from scripts.workflows.automatic_articulation.utils.process_action import process_action
import torch
from tools.curobo_planner import MotionPlanner
from tqdm import tqdm
from scripts.workflows.utils.multi_datawrapper import MultiDatawrapper
import copy
import imageio
from scripts.workflows.open_policy.utils.buffer_utils import reset_buffer, update_buffer
import numpy as np
from tools.curobo_planner import IKPlanner


class OpenVlaEvalEnv:

    def __init__(self,
                 env,
                 env_config,
                 args_cli,
                 prompt,
                 use_relative_pose=False,
                 inference_client=None,
                 render_camera=False):
        self.env = env
        self.device = env.device
        self.env_config = env_config
        self.prompt = prompt
        self.args_cli = args_cli
        self.use_relative_pose = use_relative_pose
        # if self.use_relative_pose:
        self.reset_actions = 0 * torch.rand(env.action_space.shape,
                                            device=self.device)
        self.inference_client = inference_client
        self.inference_or_not = self.env_config["params"]["Task"][
            "inference_or_not"] and self.env_config["params"]["GS_Camera"][
                "initial"]
        self.render_camera = render_camera
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
        self.init_planner()

    def init_planner(self):
        if "franka" in self.args_cli.robot_type:
            self.curo_ik_planner = IKPlanner(self.env)
            target_pose = torch.as_tensor(
                self.env_config["params"]["Task"]["init_ee_pose"]).to(
                    self.device)
            self.target_robot_jpos = self.curo_ik_planner.plan_motion(
                target_pose[:3],
                target_pose[3:7])[0].repeat_interleave(self.env.num_envs,
                                                       dim=0)

            reset_joint_pos = self.env.scene["robot"]._data.reset_joint_pos
            reset_joint_pos[:, :7] = self.target_robot_jpos[:, :7]
            self.env.scene["robot"].data.reset_joint_pos = reset_joint_pos

    def get_root_link_state(self, env, object_name):
        object = env.scene[object_name]
        return object.data.root_link_state_w

    def refer_distance(self):
        left_finger_name_state = self.get_root_link_state(
            self.env, "wx250s_left_finger_link")
        right_finger_name_state = self.get_root_link_state(
            self.env, "wx250s_right_finger_link")
        ee_pos_w = self.get_root_link_state(self.env, "wx250s_ee_gripper_link")
        object_state = self.get_root_link_state(self.env, "eggplant")

        object_ee_distance = torch.norm(object_state[:, :2] - ee_pos_w[:, :2],
                                        dim=1)

        close_enough = torch.where(object_ee_distance < 0.03, True, False)
        height_diff = object_state[:, 2] - ee_pos_w[:, 2]
        height_enough = torch.where(abs(height_diff) < 0.02, True, False)

        in_between_gripper = torch.logical_or(
            torch.logical_and(
                object_state[:, 1] > left_finger_name_state[:, 1],
                object_state[:, 1] < right_finger_name_state[:, 1]),
            torch.logical_and(
                object_state[:, 1] < left_finger_name_state[:, 1],
                object_state[:, 1] > right_finger_name_state[:, 1]))

        in_between_gripper = torch.logical_and(
            in_between_gripper,
            close_enough,
        )
        in_between_gripper = torch.logical_and(in_between_gripper,
                                               height_enough)
        return in_between_gripper

    def inference_vla(self, last_obs, image_buffer):

        if "gs_image" in last_obs["policy"].keys():
            gs_image = last_obs["policy"]["gs_image"]
        else:

            gs_image = last_obs["policy"]["rgb"]

        if self.args_cli.base_policy == "openvla":

            response = self.inference_client.step(
                self.prompt,
                gs_image[0, 0].cpu().numpy(),
                unnorm_key=self.args_cli.unnorm_key,
            )
            actions = torch.as_tensor(response["action"],
                                      device=self.device).unsqueeze(0)
            image_buffer.append(response["image"])
        elif self.args_cli.base_policy == "openpi":
            response = self.inference_client.step({
                "observation/exterior_image_1_left":
                gs_image[0, 0].cpu().numpy(),
                "observation/wrist_image_left":
                gs_image[0, 1].cpu().numpy(),
                "observation/joint_position":
                last_obs["policy"]["joint_pos"][0, :7].cpu().numpy(),
                "observation/gripper_position":
                torch.sign(
                    last_obs["policy"]["joint_pos"][0,
                                                    -1]).cpu().numpy()[None],
                "prompt":
                self.prompt,
            })
            actions = torch.as_tensor(response[0],
                                      device=self.device).unsqueeze(0)
            gs_image_reshaped = gs_image[0].permute(1, 0, 2, 3).reshape(
                224, 224 * 2, 3)

            image_buffer.append(gs_image_reshaped.cpu().numpy())

        next_obs, rewards, terminated, time_outs, extras = self.env.step(
            actions)

        return next_obs, actions, rewards, terminated, time_outs, extras, image_buffer

    def success_or_not(self, next_obs, i, reset_frame, total_frame, rewards):

        if rewards > 10 and reset_frame:
            total_frame = np.min([i + 20, total_frame])
            self.pbar.total = total_frame  # Dynamically update tqdm total
            self.pbar.refresh()  # Refresh tqdm to show the updated total
            reset_frame = False
        return reset_frame, total_frame, rewards > 10

    def step(self, last_obs, video_name=None):
        image_buffer = []
        total_frame = self.env_config["params"]["Task"]["horizon"]
        reset_frame = True

        i = 0
        reset_frame = True
        self.pbar = tqdm(total=total_frame,
                         desc="Processing frames")  # Initialize tqdm

        while i < total_frame:
            # if self.inference_or_not:
            next_obs, actions, rewards, terminated, time_outs, extras, image_buffer = self.inference_vla(
                last_obs, image_buffer)
            # else:
            #     next_obs, rewards, terminated, time_outs, extras = self.env.step(
            #         self.reset_actions)
            reset_frame, total_frame, success = self.success_or_not(
                next_obs, i, reset_frame, total_frame, rewards)

            update_buffer(self, next_obs, last_obs, actions, rewards,
                          terminated or success, time_outs)

            last_obs = copy.deepcopy(next_obs)
            i += 1
            self.pbar.update(1)  # Increment the progress bar

        success = rewards > 10
        if video_name is not None:
            self.collector_interface.save_video(
                f"{video_name}_{success.cpu().numpy()[0]}", image_buffer)

        if success:
            self.collector_interface.add_demonstraions_to_buffer(
                self.obs_buffer, self.action_buffer, self.rewards_buffer,
                self.does_buffer, self.next_obs_buffer)
        reset_buffer(self)
        return success.cpu().numpy()[0]

    def reset(self):
        self.env.reset()

        # if not self.use_relative_pose:

        #     self.env.scene["robot"]
        #     index, _ = self.env.scene["robot"].find_bodies(
        #         "wx250s_ee_gripper_link")
        #     ee_pose = self.env.scene["robot"].data.body_state_w[:,
        #                                                         index[0], :7]
        #     self.reset_actions = torch.cat(
        #         [ee_pose,
        #          torch.zeros((len(ee_pose), 1)).to(self.device)],
        #         dim=-1)

        for i in range(self.env_config["params"]["Task"]["reset_horizon"]):
            obs, rewards, terminated, time_outs, extras = self.env.step(
                self.reset_actions)

        return obs
