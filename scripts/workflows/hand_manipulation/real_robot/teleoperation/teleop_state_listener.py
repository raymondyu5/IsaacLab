import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import numpy as np

import threading
import time
from copy import deepcopy
from typing import Optional, Tuple
import zmq

import os


def kill_process_using_port(port):
    import subprocess
    try:
        output = subprocess.check_output(f"lsof -i :{port} -t",
                                         shell=True).decode().split()
        for pid in output:
            print(f"🔪 Killing process {pid} using port {port}")
            os.kill(int(pid), 9)
    except subprocess.CalledProcessError:
        pass  #


class TeleopBridgeNode(Node):

    def __init__(self, port: int, host="localhost"):
        super().__init__('teleop_bridge_node')

        if host == "localhost":
            pub_bind_to = f"tcp://*:{port}"
        else:
            pub_bind_to = f"tcp://{host}:{port}"
        self.pub_bind_to = pub_bind_to

        context = zmq.Context()
        self.pub_socket = context.socket(zmq.PUB)
        self.pub_socket.bind(self.pub_bind_to)

        print(f"✅ ZMQ Publisher bound to {self.pub_bind_to}")

        self._lock = threading.Lock()

        self.subscription = self.create_subscription(String, 'teleop_update',
                                                     self.listener_callback,
                                                     10)
        self.get_logger().info("📡 Listening on /teleop_update")

    @property
    def initialized(self):
        with self._lock:
            return self._shared_initialized

    def send_teleop_cmd(
        self,
        teleop_data,
    ):
        if self.pub_socket is not None:
            self.pub_socket.send_pyobj(teleop_data)

    def listener_callback(self, msg: String):
        try:
            data = json.loads(msg.data)

            self.send_teleop_cmd(data)

        except (json.JSONDecodeError, KeyError) as e:
            self.get_logger().error(f"❌ Failed to process message: {e}")


def main(args=None):
    rclpy.init(args=args)
    kill_process_using_port(port=5555)
    node = TeleopBridgeNode(port=5555)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("👋 Keyboard interrupt")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
