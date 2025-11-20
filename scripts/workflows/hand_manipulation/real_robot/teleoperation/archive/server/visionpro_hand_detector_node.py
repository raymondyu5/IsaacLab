#!/usr/bin/env python3
import argparse
from typing import Dict
from scripts.workflows.hand_manipulation.real_robot.teleoperation.client.streamer import VisionProStreamer
import grpc
import numpy as np
import rclpy

from pytransform3d import rotations

from scripts.workflows.hand_manipulation.real_robot.teleoperation.server.vision_detector_utils import *

import logging

logger = logging.getLogger(__name__)

import socket
import threading
import json
from typing import Optional
from threading import Thread, Lock
import time


class VisionProHandDetector:

    def __init__(self,
                 vision_pro_ip,
                 host="0.0.0.0",
                 port=10001,
                 verbose=False):
        self.vision_pro_ip = vision_pro_ip
        self.streamer: Optional[VisionProStreamer] = None
        self.latest_info = {"status": "not connected"}
        self.lock = Lock()
        self.running = True
        self.verbose = verbose
        self.host = host
        self.port = port

        # Start background threads
        Thread(target=self._connect_loop, daemon=True).start()
        Thread(target=self._start_server, daemon=True).start()

    def _connect_loop(self):
        while self.running:
            if self.streamer is None:
                try:
                    print(
                        "[Server] Attempting to connect to VisionProStreamer..."
                    )
                    self.streamer = VisionProStreamer(self.vision_pro_ip)
                    print("[Server] Connected to VisionProStreamer.")
                    Thread(target=self._update_loop, daemon=True).start()
                    return  # Exit loop once connected
                except Exception as e:
                    print(f"[Server] Connection failed: {e}")
                    with self.lock:
                        self.latest_info = {"status": "not connected"}
                    self.streamer = None

    def _update_loop(self):
        while self.running:
            try:
                if self.streamer:
                    info = self.streamer.get_latest()
                    with self.lock:
                        self.latest_info = info
            except Exception as e:
                print(f"[Server] Failed to get data: {e}")
                with self.lock:
                    self.latest_info = {"status": "stream error"}
                self.streamer = None  # Force reconnect
                Thread(target=self._connect_loop, daemon=True).start()
                break

    def _start_server(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
            server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            server_sock.bind((self.host, self.port))
            server_sock.listen(5)
            print(f"[Server] Listening on {self.host}:{self.port}")

            while self.running:
                conn, addr = server_sock.accept()
                print(f"[Server] Connected by {addr}")
                Thread(target=self._handle_client, args=(conn, ),
                       daemon=True).start()

    def _handle_client(self, conn):
        with conn:
            try:
                while True:
                    data = conn.recv(1024)
                    if not data:
                        break
                    command = data.decode("utf-8").strip()
                    if command == "get":
                        with self.lock:
                            safe_info = self._to_json_safe(self.latest_info)
                        response = json.dumps(safe_info)
                        conn.sendall(response.encode("utf-8") + b"\n")
                    else:
                        conn.sendall(b"Unknown command\n")
            except Exception as e:
                print(f"[Server] Client error: {e}")

    def _to_json_safe(self, data):
        if isinstance(data, dict):
            return {k: self._to_json_safe(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._to_json_safe(v) for v in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (np.float32, np.float64)):
            return float(data)
        elif isinstance(data, (np.int32, np.int64)):
            return int(data)
        else:
            return data


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--vision_pro_ip",
        default="10.0.0.160",
        type=str,
        help="IP address of Apple Vision Pro in a local network.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        type=str,
        help="Server host to bind (default: all interfaces)",
    )
    parser.add_argument(
        "--port",
        default=10001,
        type=int,
        help="Port to run the TCP server on.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    parser.add_argument(
        "--ros-args",
        required=False,
        action="store_true",
        help="Dirty hack to be compatible with ROS2 node action.",
    )
    args = parser.parse_args()

    # Start the detector
    detector = VisionProHandDetector(
        vision_pro_ip=args.vision_pro_ip,
        host=args.host,
        port=args.port,
        verbose=args.verbose,
    )

    print("[Main] VisionProHandDetector is running.")
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("\n[Main] Shutting down.")
        detector.running = False


if __name__ == "__main__":
    main()
