import grpc
from avp_stream.grpc_msg import *
from threading import Thread

import time
import numpy as np
from avp_stream.utils.grpc_utils import process_matrices, process_matrix
from scripts.workflows.hand_manipulation.real_robot.teleoperation.server.vision_detector_utils import *


class VisionProStreamer:

    def __init__(self, ip, record=True):

        # Vision Pro IP
        self.ip = ip
        self.record = record
        self.recording = []
        self.latest = None
        self.axis_transform = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]],
                                       dtype=np.float64)
        self.start_streaming()

    def start_streaming(self):

        stream_thread = Thread(target=self.stream)
        stream_thread.start()
        while self.latest is None:
            pass
        print(' == DATA IS FLOWING IN! ==')
        print('Ready to start streaming.')

    def stream(self):
        # Ref: https://github.com/Improbable-AI/VisionProTeleop
        request = handtracking_pb2.HandUpdate()
        try:
            with grpc.insecure_channel(f"{self.ip}:12345") as channel:
                stub = handtracking_pb2_grpc.HandTrackingServiceStub(channel)
                responses = stub.StreamHandUpdates(request)

                for response in responses:

                    left_joints = process_matrices(
                        response.left_hand.skeleton.jointMatrices)
                    right_joints = process_matrices(
                        response.right_hand.skeleton.jointMatrices)
                    left_joints = two_mat_batch_mul(left_joints,
                                                    OPERATOR2AVP_LEFT.T)
                    right_joints = two_mat_batch_mul(right_joints,
                                                     OPERATOR2AVP_RIGHT.T)

                    transformations = {
                        "left_wrist":
                        three_mat_mul(
                            self.axis_transform,
                            process_matrix(response.left_hand.wristMatrix)[0],
                            OPERATOR2AVP_LEFT,
                        ),
                        "right_wrist":
                        three_mat_mul(
                            self.axis_transform,
                            process_matrix(response.right_hand.wristMatrix)[0],
                            OPERATOR2AVP_RIGHT,
                        ),
                        "left_fingers":
                        left_joints,
                        "right_fingers":
                        right_joints,
                        "head":
                        rotate_head(
                            three_mat_mul(
                                self.axis_transform,
                                process_matrix(response.Head)[0],
                                np.eye(3),
                            )),
                    }
                    if self.record:
                        self.recording.append(transformations)
                    self.latest = transformations

        except Exception as e:
            print(f"An error occurred: {e}")
            pass

    def get_latest(self):
        return self.latest

    def get_recording(self):
        return self.recording


if __name__ == "__main__":

    streamer = VisionProStreamer(ip='10.29.230.57')
    while True:

        latest = streamer.get_latest()
        print(latest)
