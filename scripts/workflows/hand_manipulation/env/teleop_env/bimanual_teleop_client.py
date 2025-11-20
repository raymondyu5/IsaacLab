import pickle
import threading
import time
from typing import List, Tuple

import numpy as np
import zmq

from bunny_teleop.init_config import (
    InitializationConfig,
    BimanualAlignmentMode,
)
from dex_retargeting.retargeting_config import RetargetingConfig

from pathlib import Path
import yaml
import copy


class TeleopClient:

    def __init__(self,
                 port: int,
                 host="localhost",
                 right_robot_joint_names=None,
                 left_robot_joint_names=None,
                 init_ee_pose=None):

        # Socket config
        self.ctx = zmq.Context()
        if host == "localhost":
            sub_bind_to = f"tcp://localhost:{port}"
        else:
            sub_bind_to = f"tcp://{host}:{port}"
        self.sub_bind_to = sub_bind_to
        self.init_ee_pose = init_ee_pose

        # Init retargeting
        self.init_setting(right_robot_joint_names, left_robot_joint_names)
        if right_robot_joint_names is not None:
            self.add_right_hand = True
        else:
            self.add_right_hand = False
        if left_robot_joint_names is not None:
            self.add_left_hand = True
        else:
            self.add_left_hand = False

        # Thread-safe shared data
        self._lock = threading.Lock()
        self.latest_info = None
        self._shared_most_recent_ee_pose = (np.zeros(7), np.zeros(7))
        self._shared_server_started = True

        # Start background thread
        self._threads = []
        thread = threading.Thread(target=self.run, args=(0, ))
        thread.daemon = True
        thread.start()
        self._threads.append(thread)
        self.init_caliberate = False
        self._last_sleep_time = time.time()

    def init_setting(self, right_robot_joint_names, left_robot_joint_names):
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

        for cfg_dict in [left_config, right_config]:
            retargeting_config = RetargetingConfig.from_dict(
                cfg_dict["retargeting"])
            retargeting = retargeting_config.build()
            self.retargetings.append(retargeting)

        self.right_robot_reorder = np.arange(16)
        self.left_robot_joint_names = np.arange(16)

        if right_robot_joint_names is not None:
            self.right_robot_reorder = np.array([
                self.retargetings[1].optimizer.target_joint_names.index(j)
                for j in right_robot_joint_names
            ])
        if left_robot_joint_names is not None:
            self.left_robot_joint_names = np.array([
                self.retargetings[0].optimizer.target_joint_names.index(j)
                for j in left_robot_joint_names
            ])

    def update_teleop_cmd(self, message):
        try:
            cmd = pickle.loads(message[0])
            # print("📥 Received teleop message")

            right_hand_joints = cmd["right_hand_joints"]
            left_hand_joints = cmd["left_hand_joints"]

            retargeted_right_qpos = self.update_last_retargeted_qpos(
                right_hand_joints, 1)
            retargeted_left_qpos = self.update_last_retargeted_qpos(
                left_hand_joints, 0)

            with self._lock:

                if not self.init_caliberate and self.latest_info is not None:

                    self._last_sleep_time = getattr(self, "_last_sleep_time",
                                                    0)
                    now = time.time()
                    # print(cmd["right_hand_pose"][:3],
                    #       self.latest_info["right_hand_pose"][:3])
                    self._is_hand_pose_good_for_init(
                        retargeted_right_qpos, retargeted_left_qpos,
                        cmd["right_hand_pose"], cmd["left_hand_pose"],
                        self.latest_info["right_hand_pose"],
                        self.latest_info["left_hand_pose"])
                    if now - self._last_sleep_time > 0.5:  # only sleep every 2 seconds
                        time.sleep(0.1)
                        self._last_sleep_time = now
                self.latest_info = copy.deepcopy(cmd)
                self.latest_info[
                    "retargeted_right_qpos"] = retargeted_right_qpos[
                        self.right_robot_reorder]
                self.latest_info[
                    "retargeted_left_qpos"] = retargeted_left_qpos[
                        self.left_robot_joint_names]

        except Exception as e:
            print(f"❌ Failed to update teleop command: {e}")

    def _is_hand_pose_good_for_init(
        self,
        right_hand_joints: np.ndarray,
        left_hand_joints: np.ndarray,
        cur_right_hand_pose: np.ndarray,
        cur_left_hand_pose: np.ndarray,
        last_right_hand_pose: np.ndarray,
        last_left_hand_pose: np.ndarray,
    ):
        # Check whether the hand wrist is moving during initialization

        # Check whether the fingers are in a flatten way
        flat_threshold = (0.01, np.deg2rad(15))
        not_far_away_threshold = 0.05
        hand_flat_spread = True
        hand_not_far_away = True
        if self.add_left_hand:
            hand_flat_spread = hand_flat_spread & (flat_threshold[0] < np.mean(
                left_hand_joints) < flat_threshold[1])

            hand_not_far_away = hand_not_far_away & (np.linalg.norm(
                cur_left_hand_pose[:3] - last_left_hand_pose[:3])
                                                     < not_far_away_threshold)
        if self.add_right_hand:
            hand_flat_spread = hand_flat_spread & (flat_threshold[0] < np.mean(
                right_hand_joints) < flat_threshold[1])
            hand_not_far_away = hand_not_far_away & (np.linalg.norm(
                cur_right_hand_pose[:3] - last_right_hand_pose[:3])
                                                     < not_far_away_threshold)
        if not hand_not_far_away:
            print("❌ Hand is not far away from the last pose")
        if not hand_flat_spread:
            print("❌ Hand is not flat")

        return hand_flat_spread

    def get_teleop_cmd(self):
        with self._lock:
            return self.latest_info

    def get_ee_pose(self):
        with self._lock:
            return self._shared_most_recent_ee_pose

    @property
    def started(self):
        with self._lock:
            return self._shared_server_started

    def wait_for_server_start(self):
        try:
            while not self.started:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Keyboard interrupt, shutting down.")
            exit()

    def update_last_retargeted_qpos(self, joint, index):
        retargeting = self.retargetings[index]
        optimizer = retargeting.optimizer
        retargeting_type = optimizer.retargeting_type
        indices = optimizer.target_link_human_indices

        if retargeting_type == "POSITION":
            ref_value = joint[indices, :]
        else:
            origin_indices = indices[0, :]
            task_indices = indices[1, :]
            ref_value = joint[task_indices, :] - joint[origin_indices, :]

        retargeted_qpos = retargeting.retarget(ref_value)
        return retargeted_qpos

    def run(self, thread_id=0):
        print(f"🧵 Thread {thread_id} listening on {self.sub_bind_to}")
        sub_socket = self.ctx.socket(zmq.SUB)
        sub_socket.setsockopt(zmq.SUBSCRIBE, b"")
        sub_socket.connect(self.sub_bind_to)

        while True:
            try:
                msg = sub_socket.recv()
                self.update_teleop_cmd([msg])
            except Exception as e:
                print(f"❌ Error in thread {thread_id}: {e}")


if __name__ == "__main__":
    client = TeleopClient(port=5500)

    while True:
        info = client.get_teleop_cmd()
        if info is not None:
            print(info["retargeted_right_qpos"].shape)

# import zmq
# import pickle
# import pdb

# # Initialize ZMQ context and subscriber socket
# ctx = zmq.Context()
# sub_socket = ctx.socket(zmq.SUB)
# sub_socket.setsockopt(zmq.SUBSCRIBE, b"")  # Subscribe to all topics
# sub_socket.connect(
#     "tcp://localhost:5500")  # Match the publisher's bind address

# print("🔄 Waiting for teleop commands...")

# while True:
#     try:
#         msg = sub_socket.recv()
#         data = pickle.loads(msg)

#         pdb.set_trace()  # Debug breakpoint

#         print("🌀 entering publish loop")
#         print("✅ Sending teleop command")

#     except Exception as e:
#         print(f"❌ Error receiving or parsing message: {e}")
