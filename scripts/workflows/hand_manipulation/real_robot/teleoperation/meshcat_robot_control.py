import zmq
import time
from zmq.eventloop import zmqstream
import pickle
import threading

from tornado import ioloop

from scripts.workflows.hand_manipulation.BC.robot_wrapper import RobotWrapper
from scripts.workflows.hand_manipulation.real_robot.teleoperation.vision_pro_track.meshcat_robot_controller import MeshcatRobotController
import pinocchio as pin
import numpy as np
from pinocchio import SE3, Quaternion
from scripts.workflows.utils.multi_datawrapper import MultiDatawrapper
import argparse
# add argparse arguments
parser = argparse.ArgumentParser(
    description="Collect teleoperation data with a real robot.")
parser.add_argument("--log_dir", default="logs/trash")
parser.add_argument("--save_path", default="trash")
parser.add_argument(
    "--num_demos",
    default=1000,
    type=int,
)
args_cli = parser.parse_args()


def pose_to_se3(pos, quat):
    quat = Quaternion(quat)  # scalar first
    return SE3(quat.matrix(), pos)


class MeschcatRobotControl(MeshcatRobotController):

    def __init__(self,
                 port: int = 6000,
                 host: str = "localhost",
                 action_space="cartesian_position"):

        self.robot_wrapper = RobotWrapper(
            "/home/weirdlab/Documents/IsaacLab/assets/robot/franka_leap/trash/franka_right_leap_long_finger.urdf",
            viz_robot=True,
            mesh_synthesis=False,
        )
        self.init_data_buffer()

        self.collector_interface = MultiDatawrapper(
            args_cli,
            None,
            save_path=args_cli.save_path,
        )

        super().__init__(port, host, action_space)

    def init_data_buffer(self, ):

        self.obs_buffer = []
        self.actions_buffer = []
        self.does_buffer = []
        self.rewards_buffer = []

    def run_robot(self):

        robot_action, command = self.get_teleop_data()
        # if command is not None:
        #     if len(command) > 0:
        #         print(command)
        for action in robot_action:
            self.robot_wrapper.step(action)

            if command is not None:
                if "play" in command:
                    self.action_buffer.append(action)
                    return

                if "reset" in command:
                    self.init_data_buffer()
                    self.collector_interface.add_demonstraions_to_buffer(
                        self.obs_buffer,
                        self.actions_buffer,
                        self.rewards_buffer,
                        self.does_buffer,
                        external_filename="uncategorized")
                if "save" in command:
                    self.init_data_buffer()
                    self.collector_interface.add_demonstraions_to_buffer(
                        self.obs_buffer,
                        self.actions_buffer,
                        self.rewards_buffer,
                        self.does_buffer,
                        external_filename="success")
                if "remove" in command:
                    self.init_data_buffer()


if __name__ == "__main__":
    teleop_client = MeschcatRobotControl()

    while True:
        start_time = time.time()
        teleop_client.run_robot()
