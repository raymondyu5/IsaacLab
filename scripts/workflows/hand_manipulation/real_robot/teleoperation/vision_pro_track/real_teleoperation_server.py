import socket
import threading
import grpc
import time
import numpy as np

from scripts.workflows.hand_manipulation.real_robot.teleoperation.utils.visionpro_utils import *
from scripts.workflows.hand_manipulation.real_robot.teleoperation.vision_pro_track import handtracking_pb2, handtracking_pb2_grpc
from avp_stream.utils.grpc_utils import process_matrices, process_matrix

import zmq


class TeleopServer:

    def __init__(self,
                 host_ip="0.0.0.0",
                 port=8888,
                 avp_address="192.168.0.50",
                 send_command_to_robot=False):
        self.host_ip = host_ip
        self.port = port
        self.avp_address = avp_address
        self.latest_transformation = {}
        self.axis_transform = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]],
                                       dtype=np.float64)

        self.latest_command = {}
        self.latest_robot_command = {}
        self.failed = True

        if send_command_to_robot:
            self.pub_port = 6000
            kill_process_using_port(self.pub_port)

            # ✅ New: ZMQ PUB socket setup
            self.zmq_context = zmq.Context()
            self.pub_socket = self.zmq_context.socket(zmq.PUB)
            # You can change this
            self.pub_socket.bind(f"tcp://*:{self.pub_port}")
            print(f"🟣 ZMQ PUB bound to tcp://*:{self.pub_port}")

    def send_teleop_cmd(self, msg):

        self.pub_socket.send_pyobj(msg)

    def start(self):
        kill_process_using_port(self.port)

        # Start TCP server in thread
        tcp_thread = threading.Thread(target=self._start_tcp_server,
                                      daemon=True)
        tcp_thread.start()

        # Start Vision Pro streaming in thread
        grpc_thread = threading.Thread(target=self.stream_from_vision_pro,
                                       daemon=True)
        grpc_thread.start()
        time.sleep(0.01)

    def _start_tcp_server(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((self.host_ip, self.port))
        server_socket.listen()

        print(f"🟢 TCP Server listening on {self.host_ip}:{self.port}")
        while True:
            conn, addr = server_socket.accept()

            client_thread = threading.Thread(target=self.read_command,
                                             args=(conn, addr),
                                             daemon=True)
            client_thread.start()

    def get_latest_command(self):
        return self.latest_command, self.latest_transformation

    def send_command(self, command):
        send_command_to_vision_pro(self.avp_address, self.port, command)
        # print(f"📤 Sent command: {command}")

    def read_command(self, conn, addr):
        with conn:
            while True:
                try:
                    data = conn.recv(1024)
                    if not data:
                        self.latest_command = {}

                        break
                    self.latest_command = data.decode()
                    # print(f"📥 Received from {addr}: {self.latest_command}")

                except ConnectionResetError:
                    print(f"❌ Connection reset by {addr}")
                    break
                except Exception as e:
                    print(f"❌ Error with {addr}: {e}")
                    break

    def stream_from_vision_pro(self):
        request = handtracking_pb2.HandUpdate()
        while True:
            try:
                print(
                    f"🌐 Attempting to connect to Vision Pro at {self.avp_address}:12345"
                )
                with grpc.insecure_channel(
                        f"{self.avp_address}:12345") as channel:
                    stub = handtracking_pb2_grpc.HandTrackingServiceStub(
                        channel)
                    responses = stub.StreamHandUpdates(request)
                    print("✅ Connected to Vision Pro")

                    for response in responses:
                        start_time = time.time()

                        left_joints = process_matrices(
                            response.left_hand.skeleton.jointMatrices)
                        right_joints = process_matrices(
                            response.right_hand.skeleton.jointMatrices)

                        left_joints = two_mat_batch_mul(
                            left_joints, OPERATOR2AVP_LEFT.T)
                        right_joints = two_mat_batch_mul(
                            right_joints, OPERATOR2AVP_RIGHT.T)

                        right_joint_pose = joint_avp2hand(right_joints).astype(
                            np.float32)
                        left_joint_pose = joint_avp2hand(left_joints).astype(
                            np.float32)

                        self.latest_transformation = {
                            "left_wrist":
                            three_mat_mul(
                                self.axis_transform,
                                process_matrix(
                                    response.left_hand.wristMatrix)[0],
                                OPERATOR2AVP_LEFT,
                            ),
                            "right_wrist":
                            three_mat_mul(
                                self.axis_transform,
                                process_matrix(
                                    response.right_hand.wristMatrix)[0],
                                OPERATOR2AVP_RIGHT,
                            ),
                            "left_fingers":
                            left_joint_pose[:, :3, 3],
                            "right_fingers":
                            right_joint_pose[:, :3, 3],
                            "head":
                            rotate_head(
                                three_mat_mul(
                                    self.axis_transform,
                                    process_matrix(response.Head)[0],
                                    np.eye(3),
                                )),
                        }
                        print(time.time() - start_time)
                    self.failed = False

            except grpc.RpcError as e:
                self.failed = True
                self.latest_transformation = {}
                print(f"❌ gRPC error: {e}")
            except Exception as e:
                self.failed = True
                self.latest_transformation = {}
                print(f"❌ Unexpected error: {e}")
            print("🔁 Reconnecting to Vision Pro in 3 seconds...")
            if self.failed:
                time.sleep(3)
            else:

                time.sleep(0.01)


if __name__ == "__main__":
    server = TeleopServer()
    server.start()

    # Main loop (non-blocking)

    while True:
        start_time = time.time()
        _, pose = server.get_latest_command()
        # print(time.time() - start_time)
