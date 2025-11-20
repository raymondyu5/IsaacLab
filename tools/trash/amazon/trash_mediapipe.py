import cv2
import mediapipe as mp
import numpy as np
import zmq
import json
import time
from typing import Optional


class MediaPipeServer:

    def __init__(self, port: int, host="localhost"):
        self.ctx = zmq.Context()
        self.pub_socket: Optional[zmq.Socket] = self.ctx.socket(zmq.PUB)

        if host == "localhost":
            self.pub_bind_to = f"tcp://*:{port}"
        else:
            self.pub_bind_to = f"tcp://{host}:{port}"

        self.pub_socket.bind(self.pub_bind_to)
        print(f"🛰️ MediaPipe ZMQ Server publishing on {self.pub_bind_to}")

    def send_landmarks(self, landmarks: list):
        message = json.dumps({
            "timestamp": time.time(),
            "landmarks": landmarks
        })
        self.pub_socket.send_string(message)


def run_hand_tracking(server: MediaPipeServer):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    cap = cv2.VideoCapture(0)

    JOINT_NAMES = [
        "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
        "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP",
        "INDEX_FINGER_TIP", "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP",
        "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP", "RING_FINGER_MCP",
        "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP", "PINKY_MCP",
        "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
    ]

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    coords = [[lm.x, lm.y, lm.z]
                              for lm in hand_landmarks.landmark]
                    server.send_landmarks(coords)

    except KeyboardInterrupt:
        print("🛑 Stopped by user.")
    finally:
        cap.release()


if __name__ == "__main__":
    server = MediaPipeServer(port=5555)
    run_hand_tracking(server)
