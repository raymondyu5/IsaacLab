import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import numpy as np


class TeleopStatePublisher():

    def __init__(self):
        if not rclpy.ok():
            rclpy.init()

        self.ros_node = rclpy.create_node('teleop_state_client')
        self.teleop_publisher = self.ros_node.create_publisher(
            String, 'teleop_update', 10)

    def publish_to_ros(
        self,
        action_dict,
        reset_teleoperation: bool,
        teleoperation_active: bool,
        save_teleoperation_data: bool,
        remove_teleoperation_data: bool,
        replay_teleoperation_active: bool,
        init_ee_pose: np.ndarray = None,
        init_arm_qpos: np.ndarray = None,
    ):
        msg = String()

        # Convert np.ndarray to list for serialization
        safe_action_dict = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in action_dict.items()
        }

        payload = {
            "reset_teleoperation":
            reset_teleoperation,
            "teleoperation_active":
            teleoperation_active,
            "save_teleoperation_data":
            save_teleoperation_data,
            "remove_teleoperation_data":
            remove_teleoperation_data,
            "replay_teleoperation_active":
            replay_teleoperation_active,
            "init_ee_pose":
            init_ee_pose.cpu().numpy().tolist(),
            "init_arm_qpos":
            init_arm_qpos.cpu().numpy().tolist()
            if init_arm_qpos is not None else None
        }
        payload.update(safe_action_dict)

        msg.data = json.dumps(payload)
        self.teleop_publisher.publish(msg)
