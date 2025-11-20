from scripts.workflows.hand_manipulation.utils.cloudxr.leap_retargeter import LeapRetargeter

from scripts.workflows.hand_manipulation.utils.cloudxr.arm_hand_wrapper import ArmHandWrapper

from scripts.workflows.utils.multi_datawrapper import MultiDatawrapper
import torch
from isaaclab.devices import OpenXRDevice

from isaaclab.devices.openxr import XrCfg

import isaaclab.utils.math as math_utils

import copy
import socket
import json

import yaml


def send_command_to_vision_pro(ip_address: str, port: int, command: str):

    data = command
    json_data = json.dumps(data).encode("utf-8")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((ip_address, port))
        s.sendall(json_data)
        print(f"✅ Sent command to Vision Pro: {command}")


class CommandWrapper(ArmHandWrapper):

    def __init__(self, env, args_cli, env_cfg):
        self.args_cli = args_cli
        self.env_cfg = env_cfg
        self.collector_interface = None
        self.env = env

        self.init_settings()
        super().__init__(self.env, self.teleop_interface, self.args_cli,
                         self.env_cfg)
        if self.args_cli.save_path is not None:
            self.collector_interface = MultiDatawrapper(
                args_cli,
                env_cfg,
                save_path=args_cli.save_path,
            )
            # self.collector_interface.init_collector_interface()
            self.init_data_buffer()
        self.avp_ip = args_cli.avp_ip
        self.port = 5555

    def init_data_buffer(self, ):

        self.obs_buffer = []
        self.actions_buffer = []
        self.does_buffer = []
        self.rewards_buffer = []

    def init_settings(self):

        self.reset_stopped_teleoperation = False
        self.teleoperation_active = False
        self.device = self.env.unwrapped.device

        retarger = LeapRetargeter(
            self.env,
            self.args_cli.add_left_hand,
            self.args_cli.add_right_hand,
            enable_visualization=True,
            num_open_xr_hand_joints=52,
            device=self.env.unwrapped.device,
            use_arm=True
            if self.env_cfg["params"]["arm_type"] is not None else False,
        )

        with open(
                f"source/config/task/hand_env/teleoperation/bunny/teleop_user_info/teleop.yml",
                'r') as file:
            user_config = yaml.safe_load(file)

        anchor_pos = user_config["xrcfg_xyz"]
        anchor_rot = user_config["xrcfg_quat"]

        self.teleop_interface = OpenXRDevice(
            XrCfg(anchor_pos=anchor_pos, anchor_rot=anchor_rot), [retarger])
        self.reset_stopped_teleoperation = False
        self.teleoperation_active = False
        self.change_reset_objects = False
        self.replay_teleoperation_active = False
        self.init = False
        self.old_obs_buffer = []
        self.old_actions_buffer = []

        self.teleop_interface.add_callback("RESET",
                                           self.reset_recording_instance)
        self.teleop_interface.add_callback("START", self.start_teleoperation)
        self.teleop_interface.add_callback("STOP", self.stop_teleoperation)
        self.teleop_interface.add_callback("REPLAY", self.replay_teleoperation)

        self.teleop_interface.add_callback("SAVE", self.save_teleoperation)
        self.teleop_interface.add_callback("REMOVE", self.remove_teleoperation)
        self.teleop_interface.add_callback("CHANGE Object",
                                           self.change_objects)
        self.teleop_interface.add_callback("USER", self.handle_user_confirm)

    # Callback handlers
    def reset_recording_instance(self):

        if self.collector_interface is not None:
            send_command_to_vision_pro(
                ip_address=self.avp_ip,
                port=self.port,
                command="Save teleop, You totally collected {} data points.".
                format(self.collector_interface.traj_count))

        print("Saving teleoperation data", len(self.obs_buffer))
        if len(self.obs_buffer) > 40:

            self.old_obs_buffer = copy.deepcopy(self.obs_buffer)
            self.old_actions_buffer = copy.deepcopy(self.actions_buffer)

            if self.collector_interface is not None:
                self.collector_interface.add_demonstraions_to_buffer(
                    self.obs_buffer,
                    self.actions_buffer,
                    self.rewards_buffer,
                    self.does_buffer,
                )
        self.init_data_buffer(save_old_data=True)

        self.reset_stopped_teleoperation = True

    def handle_user_confirm(self, message=None):
        print("✅ User confirm received.", message)

        self.teleop_user = message
        self.init_settings()

    def change_objects(self):
        self.change_reset_objects = True

    def start_teleoperation(self):
        send_command_to_vision_pro(ip_address=self.avp_ip,
                                   port=self.port,
                                   command="start teleop")

        self.teleoperation_active = True

    def replay_teleoperation(self):
        send_command_to_vision_pro(ip_address=self.avp_ip,
                                   port=self.port,
                                   command="replay teleop")

        self.replay_teleoperation_active = True

    def stop_teleoperation(self):

        self.teleoperation_active = False

    def save_teleoperation(self):

        if self.collector_interface is not None:
            send_command_to_vision_pro(
                ip_address=self.avp_ip,
                port=self.port,
                command=
                f"Save teleop, You totally collected {self.collector_interface.traj_count} demo with length {len(self.obs_buffer)}."
            )
        print("Saving teleoperation data", len(self.obs_buffer))
        if len(self.obs_buffer) > 40:

            self.old_obs_buffer = copy.deepcopy(self.obs_buffer)
            self.old_actions_buffer = copy.deepcopy(self.actions_buffer)

            if self.collector_interface is not None:
                self.collector_interface.add_demonstraions_to_buffer(
                    self.obs_buffer,
                    self.actions_buffer,
                    self.rewards_buffer,
                    self.does_buffer,
                )
        self.init_data_buffer(save_old_data=True)

    def remove_teleoperation(self):
        send_command_to_vision_pro(ip_address=self.avp_ip,
                                   port=self.port,
                                   command="Remove Collect Data".format(
                                       len(self.obs_buffer)))
        self.init_data_buffer(save_old_data=True)
        self.reset_stopped_teleoperation = True
