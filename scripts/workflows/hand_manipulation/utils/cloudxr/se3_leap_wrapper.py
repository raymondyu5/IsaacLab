from scripts.workflows.hand_manipulation.utils.cloudxr.leap_retargeter import LeapRetargeter

from scripts.workflows.hand_manipulation.utils.cloudxr.arm_hand_wrapper import ArmHandWrapper

from scripts.workflows.utils.multi_datawrapper import MultiDatawrapper
import torch
from isaaclab.devices import OpenXRDevice

from isaaclab.devices.openxr import XrCfg

import isaaclab.utils.math as math_utils
import yaml
import copy
from scripts.workflows.hand_manipulation.utils.cloudxr.retarget_utils import VisionProClient


class Se3LeapWrapper(ArmHandWrapper):

    def __init__(
        self,
        env,
        env_cfg,
        args_cli,
    ):
        self.args_cli = args_cli
        self.env_cfg = env_cfg
        self.collector_interface = None
        self.env = env

        self.init_settings()
        self.avp_ip = args_cli.avp_ip
        self.port = 5555
        self.visionpro_client = VisionProClient(self.avp_ip, self.port)
        super().__init__(self.env, self.teleop_interface, self.args_cli,
                         self.env_cfg, self.teleop_config,
                         self.visionpro_client)
        if self.args_cli.save_path is not None:
            self.collector_interface = MultiDatawrapper(
                args_cli,
                env_cfg,
                save_path=args_cli.save_path,
            )
            # self.collector_interface.init_collector_interface()
            self.init_data_buffer()

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
        )

        with open(
                f"source/config/task/hand_env/teleoperation/bunny/teleop_user_info/teleop.yml",
                'r') as file:
            self.teleop_config = yaml.safe_load(file)

        anchor_pos = self.teleop_config["xrcfg_xyz"]
        anchor_rot = self.teleop_config["xrcfg_quat"]

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
            self.visionpro_client.send_command(
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
        self.init_data_buffer()

        self.reset_stopped_teleoperation = True

    def handle_user_confirm(self, message=None):
        print("✅ User confirm received.", message)

        self.teleop_user = message
        self.init_settings()

    def change_objects(self):
        self.change_reset_objects = True

    def start_teleoperation(self):
        self.visionpro_client.send_command(command="start teleop")

        self.teleoperation_active = True

    def replay_teleoperation(self):
        self.visionpro_client.send_command(command="replay teleop")

        self.replay_teleoperation_active = True

    def stop_teleoperation(self):

        self.teleoperation_active = False

    def save_teleoperation(self):

        if self.collector_interface is not None:
            self.visionpro_client.send_command(
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
        self.init_data_buffer()

    def remove_teleoperation(self):
        self.visionpro_client.send_command(
            command="Remove Collect Data".format(len(self.obs_buffer)))
        self.init_data_buffer()
        self.reset_stopped_teleoperation = True

    def step(self):
        """
        Step the Leap object to get the current hand pose.
        """

        if self.replay_teleoperation_active:
            print("Replay teleoperation", len(self.old_actions_buffer))

            self.replay_teleoperation_data(
                self.old_obs_buffer,
                self.old_actions_buffer,
            )
            self.replay_teleoperation_active = False
            self.teleoperation_active = True

        obs, rewards, done, extras, actions, reset, change_obejct = self.step_teleoperation(
            self.teleoperation_active, self.reset_stopped_teleoperation,
            self.change_reset_objects)

        if obs is not None and self.collector_interface is not None:

            self.obs_buffer.append(obs)
            self.actions_buffer.append(actions)
            self.does_buffer.append(done)
            self.rewards_buffer.append(rewards)
        if reset:
            self.reset_stopped_teleoperation = False
            self.init_data_buffer()
        if change_obejct:
            self.change_reset_objects = False
