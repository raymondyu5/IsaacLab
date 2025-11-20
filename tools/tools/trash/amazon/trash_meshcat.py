import pinocchio as pin
import numpy as np
import socket
import json
import time
from os.path import join
from pinocchio.visualize import MeshcatVisualizer


class HandVisualizerClient:

    def __init__(self,
                 urdf_path,
                 model_path,
                 server_host='localhost',
                 server_port=9999):
        # Load model
        self.model, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(
            urdf_path, model_path, pin.JointModelFreeFlyer())
        self.q0 = pin.neutral(self.model)
        self.server_host = server_host
        self.server_port = server_port

        # Setup 3 visualizers
        self.viz_main = MeshcatVisualizer(self.model, self.collision_model,
                                          self.visual_model)
        self.viz_main.initViewer(open=True)
        self.viz_main.loadViewerModel(rootNodeName="raw_hand")
        self.viz_main.display(self.q0)

        self.viz_vae = MeshcatVisualizer(self.model, self.collision_model,
                                         self.visual_model)
        self.viz_vae.initViewer(self.viz_main.viewer)
        self.viz_vae.loadViewerModel(rootNodeName="vae_hand")
        self._display_offset(self.viz_vae, y_offset=0.3)

        self.viz_pca = MeshcatVisualizer(self.model, self.collision_model,
                                         self.visual_model)
        self.viz_pca.initViewer(self.viz_main.viewer)
        self.viz_pca.loadViewerModel(rootNodeName="pca_hand")
        self._display_offset(self.viz_pca, y_offset=0.6)

    def _display_offset(self, viz, y_offset):
        q = self.q0.copy()
        q[1] += y_offset
        viz.display(q)

    def _get_slider_values(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect((self.server_host, self.server_port))
                sock.sendall(b"get_values\n")
                data = sock.recv(4096)
                return json.loads(data.decode())
        except Exception as e:
            print("[Socket error]", e)
            return {
                "raw": [0.0] * 16,
                "vae": [],
                "pca": [],
                "control": [0.0] * 16
            }

    def run(self, frequency_hz=30):
        dt = 1.0 / frequency_hz
        while True:
            values = self._get_slider_values()
            num_joints = self.model.nq - 7

            # === RAW update ===
            raw = values.get("raw", [0.0] * num_joints)

            time.sleep(dt)


if __name__ == "__main__":
    model_path = "source/assets/robot/franka/urdf/franka_description/robots"
    urdf_filename = "panda_arm_hand.urdf"
    urdf_path = join(model_path, urdf_filename)
    import pdb
    pdb.set_trace()

    visualizer = HandVisualizerClient(urdf_path=urdf_path,
                                      model_path=model_path,
                                      server_host="localhost",
                                      server_port=9999)
    visualizer.run()
