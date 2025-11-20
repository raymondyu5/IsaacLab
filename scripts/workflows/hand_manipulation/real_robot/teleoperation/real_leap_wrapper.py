import torch
from isaaclab.devices import OpenXRDevice

from isaaclab.devices.openxr import XrCfg

import isaaclab.utils.math as math_utils
from scripts.workflows.hand_manipulation.utils.cloudxr.leap_retargeter import LeapRetargeter
from scripts.workflows.hand_manipulation.real_robot.teleoperation.real_arm_retargeter import RealArmRetargeter
from scripts.workflows.utils.multi_datawrapper import MultiDatawrapper
import copy
import socket
import json
import yaml

from scripts.workflows.hand_manipulation.real_robot.teleoperation.teleop_server import TeleopServer


def send_command_to_vision_pro(ip_address: str, port: int, command: str):

    data = command
    json_data = json.dumps(data).encode("utf-8")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((ip_address, port))
        s.sendall(json_data)
        # print(f"✅ Sent command to Vision Pro: {command}")


class RealLeapWrapper(TeleopServer):
    """
    Wrapper for the SE3Leap class to handle the Leap Motion data and perform necessary transformations.
    """

    def __init__(self,
                 env,
                 env_cfg,
                 args_cli,
                 teleop_user="Entong",
                 avp_ip="10.0.0.160",
                 port=5555,
                 init_hand_dir="face_down"):
        """
        Initialize the Se3LeapWrapper with a Leap object.

        Args:
            leap: An instance of the Leap class.
        """
        self.env = env
        self.env_cfg = env_cfg
        self.args_cli = args_cli
        self.teleop_user = teleop_user
        self.port = port  # Port for Vision Pro communication
        self.init_hand_dir = init_hand_dir

        self.init_settings()
        if args_cli.save_path is None:

            self.collector_interface = MultiDatawrapper(
                args_cli,
                env_cfg,
                save_path=args_cli.save_path,
            )
        else:
            self.collector_interface = None

        self.init_data_buffer()

        self.avp_ip = avp_ip

        super().__init__()

        send_command_to_vision_pro(ip_address=self.avp_ip,
                                   port=self.port,
                                   command="initialize teleop finished")

    def init_data_buffer(self, ):

        self.obs_buffer = []
        self.actions_buffer = []
        self.does_buffer = []
        self.rewards_buffer = []

    def init_settings(self):

        self.device = self.env.unwrapped.device

        retarger = LeapRetargeter(
            self.env,
            self.args_cli.add_left_hand,
            self.args_cli.add_right_hand,
            teleop_user=self.teleop_user,
            enable_visualization=True,
            num_open_xr_hand_joints=13,
            device=self.env.unwrapped.device,
            use_arm=True
            if self.env_cfg["params"]["arm_type"] is not None else False,
        )

        try:
            with open(
                    f"source/config/task/hand_env/teleoperation/bunny/teleop_user_info/{self.teleop_user}.yml",
                    'r') as file:
                self.user_config = yaml.safe_load(file)

        except:
            with open(
                    f"source/config/task/hand_env/teleoperation/bunny/teleop_user_info/Entong.yml",
                    'r') as file:
                self.user_config = yaml.safe_load(file)

        anchor_pos = self.user_config["xrcfg_xyz"]
        anchor_rot = self.user_config["xrcfg_quat"]
        self.teleop_interface = OpenXRDevice(
            XrCfg(anchor_pos=anchor_pos, anchor_rot=anchor_rot), [retarger])
        self.should_reset_teleoperation = False
        self.teleoperation_active = False

        self.save_teleoperation_data = True
        self.remove_teleoperation_data = False

        self.replay_teleoperation_active = False

        self.init = False
        self.old_obs_buffer = []
        self.old_actions_buffer = []

        self.teleop_interface.add_callback("RESET", self.reset_teleoperation)
        self.teleop_interface.add_callback("START", self.start_teleoperation)
        self.teleop_interface.add_callback("STOP", self.stop_teleoperation)
        self.teleop_interface.add_callback("REPLAY", self.replay_teleoperation)

        self.teleop_interface.add_callback("SAVE", self.save_teleoperation)
        self.teleop_interface.add_callback("REMOVE", self.remove_teleoperation)
        self.teleop_interface.add_callback("CHANGE Object",
                                           self.change_objects)
        self.teleop_interface.add_callback("USER", self.handle_user_confirm)

        self.real_arm_retarget = RealArmRetargeter(
            self.env,
            self.teleop_interface,
            self.args_cli,
            self.env_cfg,
            teleop_config=self.user_config,
            init_hand_dir=self.init_hand_dir)
        self.step_function = self.real_arm_retarget.step_teleoperation

    # Callback handlers
    def reset_teleoperation(self):
        self.should_reset_teleoperation = True
        print("✅ Teleoperation reset requested.",
              self.should_reset_teleoperation)
        print(len(self.obs_buffer), len(self.actions_buffer))
        if len(self.obs_buffer) > 0 and self.collector_interface is not None:

            self.collector_interface.add_demonstraions_to_buffer(
                self.obs_buffer,
                self.actions_buffer,
                self.rewards_buffer,
                self.does_buffer,
                external_filename="unknown")
        send_command_to_vision_pro(ip_address=self.avp_ip,
                                   port=self.port,
                                   command="reset teleop")

    def handle_user_confirm(self, message=None):
        print("✅ User confirm received.", message)

        self.teleop_user = message
        self.init_settings()

    def change_objects(self):
        pass

    def start_teleoperation(self):
        self.teleoperation_active = True

        send_command_to_vision_pro(ip_address=self.avp_ip,
                                   port=self.port,
                                   command="start teleop")

    def replay_teleoperation(self):
        self.replay_teleoperation_active = True

        send_command_to_vision_pro(ip_address=self.avp_ip,
                                   port=self.port,
                                   command="replay teleop")

    def stop_teleoperation(self):
        self.teleoperation_active = False

        send_command_to_vision_pro(ip_address=self.avp_ip,
                                   port=self.port,
                                   command="stop teleop")

    def save_teleoperation(self):

        self.old_obs_buffer = copy.deepcopy(self.obs_buffer)
        self.old_actions_buffer = copy.deepcopy(self.actions_buffer)

        self.collector_interface.add_demonstraions_to_buffer(
            self.obs_buffer,
            self.actions_buffer,
            self.rewards_buffer,
            self.does_buffer,
            external_filename="success")

        self.init_data_buffer()
        self.save_teleoperation_data = True
        send_command_to_vision_pro(
            ip_address=self.avp_ip,
            port=5555,
            command="Save teleop, You totally collected {} data points.".
            format(self.collector_interface.traj_count))
        print("Saving teleoperation data", len(self.obs_buffer))

    def remove_teleoperation(self):
        self.collector_interface.add_demonstraions_to_buffer(
            self.obs_buffer,
            self.actions_buffer,
            self.rewards_buffer,
            self.does_buffer,
            external_filename="failed")
        self.init_data_buffer()

        self.remove_teleoperation_data = True
        send_command_to_vision_pro(ip_address=self.avp_ip,
                                   port=5555,
                                   command="Remove Collect Data".format(
                                       len(self.obs_buffer)))

    def step(self):
        """
        Step the Leap object to get the current hand pose.
        """

        obs, actions, reset, intructions = self.step_function(
            self.teleoperation_active,
            self.should_reset_teleoperation,
        )

        if obs is not None:

            self.obs_buffer.append(obs | {"policy": actions})
            self.actions_buffer.append(actions)

        self.publish_once(actions, self.should_reset_teleoperation,
                          self.teleoperation_active,
                          self.save_teleoperation_data,
                          self.remove_teleoperation_data,
                          self.replay_teleoperation_active,
                          self.real_arm_retarget.init_ee_pose,
                          self.real_arm_retarget.init_arm_qpos)
        self.save_teleoperation_data = False
        self.remove_teleoperation_data = False

        self.replay_teleoperation_active = False
        if reset:
            self.should_reset_teleoperation = False
        if intructions is not None:
            send_command_to_vision_pro(ip_address=self.avp_ip,
                                       port=self.port,
                                       command=intructions)
