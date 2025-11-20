import rclpy
from rclpy.node import Node
from hand_msgs.msg import BimanualHandDetection  # Make sure this is built and sourced!

from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
import torch
from pathlib import Path
from dex_retargeting.retargeting_config import RetargetingConfig
import yaml
from bunny_teleop.init_config import BimanualAlignmentMode
from geometry_msgs.msg import Pose
import numpy as np
from threading import Lock

from scripts.workflows.hand_manipulation.env.teleop_env.bimanual_teleop_server import TeleopServer


def ros_pose_to_pos_quat(msg: Pose):
    pos = msg.position
    pos = np.array([pos.x, pos.y, pos.z])
    quat = msg.orientation
    quat = np.array([quat.w, quat.x, quat.y, quat.z])
    return np.concatenate([pos, quat])


class RealTimeHandSubscriber(Node):

    def __init__(self,
                 detection_topic_name="/visionpro/bimanual_hand_detection"):
        super().__init__('real_time_hand_listener')
        self.latest_detection = None

        self._detection_sub_group = MutuallyExclusiveCallbackGroup()
        self.detection_sub = self.create_subscription(
            BimanualHandDetection,
            f"{detection_topic_name}/results",
            self.on_hand_detection,
            qos_profile=10,
            callback_group=self._detection_sub_group,
        )
        self.initialized = False

        self.left_hand_pose = None
        self.right_hand_pose = None
        self.left_hand_joints = None
        self.right_hand_joints = None
        self.head_pose = None

        self.init_setting()
        teleop_port = 5500
        self.kill_port(teleop_port)
        self.kill_port(5501)
        teleop_host = "localhost"
        self.teleop_server = TeleopServer(teleop_port, teleop_host)
        self.teleop_server.set_initialized()

        # Server publisher
        self._teleop_publish_group = MutuallyExclusiveCallbackGroup()
        self.publish_dt = 1 / 60
        self.publish_timer = self.create_timer(
            self.publish_dt,
            self.publish_periodically,
            callback_group=self._teleop_publish_group,
        )
        self.publish_timer.reset()

    def init_setting(self):

        import os

        current_path = os.getcwd()
        urdf_dir = f"{current_path}/source/"

        RetargetingConfig.set_default_urdf_dir(urdf_dir)
        self.retargetings = []
        kinematics_path = Path(
            "source/config/task/hand_env/teleoperation/bunny/kinematics_config/bimanual_leap.yml"
        )

        with kinematics_path.open("r") as f:
            yaml_config = yaml.load(f, Loader=yaml.FullLoader)
            left_config = yaml_config["left"]
            right_config = yaml_config["right"]

        for i, cfg_dict in enumerate([left_config, right_config]):
            retargeting_config = RetargetingConfig.from_dict(
                cfg_dict["retargeting"])

            retargeting = retargeting_config.build()
            self.retargetings.append(retargeting)

    def kill_port(self, port):
        import socket
        import os
        import subprocess
        result = subprocess.run(["lsof", "-ti", f"tcp:{port}"],
                                stdout=subprocess.PIPE)
        for pid in result.stdout.decode().splitlines():
            try:
                os.kill(int(pid), 9)
            except:
                pass

    def on_hand_detection(self, data: BimanualHandDetection):
        """
        Callback function when receive hand detection results from hand detection node
        At hand monitor level, this function update the wrist pose and hand joints.

        :param data: BimanualHandDetection message data from detection node
        """

        is_success = data.detected
        if not is_success:
            return

        left_global_pose = data.left_wrist_pose
        right_global_pose = data.right_wrist_pose
        head_global_pose = data.head_pose
        self.left_hand_pose = ros_pose_to_pos_quat(left_global_pose)
        self.right_hand_pose = ros_pose_to_pos_quat(right_global_pose)
        self.left_hand_joints = np.reshape(data.left_joints, [21, 3])
        self.right_hand_joints = np.reshape(data.right_joints, [21, 3])
        self.head_pose = ros_pose_to_pos_quat(head_global_pose)

    def publish_periodically(self):

        self.teleop_server.send_teleop_cmd(self.left_hand_pose,
                                           self.left_hand_joints,
                                           self.right_hand_pose,
                                           self.right_hand_joints,
                                           self.head_pose)

    def callback(self, msg):
        print("==== New Hand Detection Msg ====")
        print(f"Head Position: {msg.head_pose.position}")
        print(f"Left Wrist Position: {msg.left_wrist_pose.position}")
        print(f"Right Wrist Position: {msg.right_wrist_pose.position}")
        print(f"Left Joints (first 5): {msg.left_joints[:15]}")
        print(f"Right Joints (first 5): {msg.right_joints[:15]}")
        print("--------------------------------")


if __name__ == "__main__":
    rclpy.init()
    node = RealTimeHandSubscriber()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
# publisher.py
# import zmq, pickle, time, numpy as np

# ctx = zmq.Context()
# pub = ctx.socket(zmq.PUB)
# pub.bind("tcp://*:5500")  # This MUST match your sub.connect()

# time.sleep(1)  # 🔥 Important: gives time for SUB to connect

# while True:
#     msg = {
#         "target_qpos": [np.random.rand(7),
#                         np.random.rand(7)],
#         "ee_pose": [np.random.rand(7), np.random.rand(7)],
#     }
#     print("📤 sending")
#     pub.send(pickle.dumps(msg))
#     time.sleep(1)
