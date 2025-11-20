import zmq
import threading
import time
import pickle

import numpy as np
import torch
import subprocess
import os
import signal


def kill_process_using_port(port):
    try:
        result = subprocess.check_output(f"lsof -i :{port} -t", shell=True)
        pids = result.decode().strip().split("\n")
        for pid in pids:
            print(f"🔪 Killing PID {pid} on port {port}")
            os.kill(int(pid), signal.SIGKILL)
    except subprocess.CalledProcessError:
        print(f"✅ No process found on port {port}")


class TeleopServer:

    def __init__(self, port=12345):

        kill_process_using_port(port)
        self.ctx = zmq.Context()
        self.pub_socket = self.ctx.socket(zmq.PUB)
        self.pub_socket.bind(f"tcp://*:{port}")
        self.running = False

    def start(self, data_provider_fn):
        """ Start the loop with a function that returns the next payload data """
        self.running = True
        self.thread = threading.Thread(target=self._loop,
                                       args=(data_provider_fn, ))
        self.thread.start()

    def _loop(self, data_provider_fn):
        while self.running:
            args = data_provider_fn()
            self.publish_once(*args)
            time.sleep(0.005)

    def publish_once(
        self,
        action_dict,
        reset_teleoperation: bool,
        teleoperation_active: bool,
        save_teleoperation_data: bool,
        remove_teleoperation_data: bool,
        replay_teleoperation_active: bool,
        init_ee_pose: torch.Tensor,
        init_arm_qpos: torch.Tensor,
    ):
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

        serialized = pickle.dumps(payload)
        self.pub_socket.send(serialized)

        time.sleep(0.005)

    def stop(self):
        self.running = False
        self.thread.join()
        self.pub_socket.close()
        self.ctx.term()
        print("🛑 ZMQ Server stopped.")


# Run the server
if __name__ == "__main__":
    server = TeleopServer(port=5555)
    try:
        server.start()
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        server.stop()
