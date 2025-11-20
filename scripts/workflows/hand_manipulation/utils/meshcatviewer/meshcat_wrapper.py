import pinocchio as pin
import numpy as np
import socket
import json
import time
from os.path import join
from pinocchio.visualize import MeshcatVisualizer
import copy
from scripts.workflows.hand_manipulation.utils.vae.data_normalizer import (
    extract_finger_joints, temporal_ensemble_finger_joints)


class HandVisualizerClient:

    def __init__(self,
                 urdf_path,
                 model_path,
                 server_host='localhost',
                 server_port=9999,
                 num_joints=16,
                 num_vae=8,
                 spacing=0.3):
        # Load model
        self.num_joints = num_joints
        self.model, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(
            urdf_path, model_path, pin.JointModelFreeFlyer())
        self.q0 = pin.neutral(self.model)
        self.server_host = server_host
        self.server_port = server_port
        self.num_vae = num_vae
        self.spacing = spacing
        self.init_raw_vizualizer()
        self.init_demo_vizualizer()
        self.init_pca_vizualizer()
        self.init_vae_vizualizer()
        self.init_reconstructed_vae_vizualizer()

        self.raw_joint_names = [
            'j1', 'j0', 'j2', 'j3', 'j12', 'j13', 'j14', 'j15', 'j5', 'j4',
            'j6', 'j7', 'j9', 'j8', 'j10', 'j11'
        ]
        self.isaac_joint_names = [
            'j1', 'j12', 'j5', 'j9', 'j0', 'j13', 'j4', 'j8', 'j2', 'j14',
            'j6', 'j10', 'j3', 'j15', 'j7', 'j11'
        ]
        self.retarget2pin = [
            self.isaac_joint_names.index(name) for name in self.raw_joint_names
        ]
        self.joint_limits = np.array(
            [[-0.314, 2.23], [-0.349, 2.094], [-0.314, 2.23], [-0.314, 2.23],
             [-1.047, 1.047], [-0.46999997, 2.4429998], [-1.047, 1.047],
             [-1.047, 1.047], [-0.5059999, 1.8849999], [-1.2, 1.8999999],
             [-0.5059999, 1.8849999], [-0.5059999, 1.8849999],
             [-0.366, 2.0419998], [-1.34, 1.8799999], [-0.366, 2.0419998],
             [-0.366, 2.0419998]],
            dtype=np.float32)[self.retarget2pin]

    def init_raw_vizualizer(self):

        # Setup 3 visualizers
        self.viz_main = MeshcatVisualizer(self.model, self.collision_model,
                                          self.visual_model)
        self.viz_main.initViewer(open=True)
        self.viz_main.loadViewerModel(rootNodeName="raw_hand")
        self.viz_main.display(self.q0)

    def init_demo_vizualizer(self):

        # demo visualizer

        self.viz_demo = MeshcatVisualizer(self.model, self.collision_model,
                                          self.visual_model)
        self.viz_demo.initViewer(self.viz_main.viewer)
        self.viz_demo.loadViewerModel(rootNodeName="demo_hand")
        self.q0_demo = self._display_offset(self.viz_demo, offset=[0, -0.3, 0])

    def init_pca_vizualizer(self):
        # PCA visualizer

        self.viz_pca = MeshcatVisualizer(self.model, self.collision_model,
                                         self.visual_model)
        self.viz_pca.initViewer(self.viz_main.viewer)
        self.viz_pca.loadViewerModel(rootNodeName="pca_hand")
        self.q0_pca = self._display_offset(self.viz_pca, offset=[0, 0.3, 0])

    def init_vae_vizualizer(self):

        # VAE visualizer (static or not used)
        self.viz_vae = []
        self.q0_vae = []

        offsets_y = [(i - (self.num_vae - 1) / 2) * self.spacing
                     for i in range(self.num_vae)]

        for i in range(self.num_vae):
            viz_vae = MeshcatVisualizer(self.model, self.collision_model,
                                        self.visual_model)
            viz_vae.initViewer(self.viz_main.viewer)
            viz_vae.loadViewerModel(rootNodeName=f"vae_hand_{i}")
            q0_vae = self._display_offset(viz_vae,
                                          offset=[0.0, offsets_y[i], 0.4])
            self.q0_vae.append(copy.deepcopy(q0_vae))
            self.viz_vae.append(viz_vae)
            del viz_vae

    def init_reconstructed_vae_vizualizer(self):

        # VAE visualizer (static or not used)
        self.reconstructed_viz_vae = []
        self.raw_viz_vae = []
        self.reconstructed_q0_vae = []
        self.raw_q0_vae = []

        offsets_y = [(i - (self.num_vae - 1) / 2) * self.spacing
                     for i in range(self.num_vae)]

        for i in range(self.num_vae):
            viz_vae = MeshcatVisualizer(self.model, self.collision_model,
                                        self.visual_model)
            viz_vae.initViewer(self.viz_main.viewer)
            viz_vae.loadViewerModel(rootNodeName=f"reconstructed_vae_hand_{i}")
            reconstructed_q0_vae = self._display_offset(
                viz_vae, offset=[-0.0, offsets_y[i], 1.0])
            self.reconstructed_q0_vae.append(
                copy.deepcopy(reconstructed_q0_vae))
            self.reconstructed_viz_vae.append(viz_vae)
            del viz_vae

            viz_vae = MeshcatVisualizer(self.model, self.collision_model,
                                        self.visual_model)
            viz_vae.initViewer(self.viz_main.viewer)

            viz_vae.loadViewerModel(rootNodeName=f"raw_vae_hand_{i}")
            raw_q0_vae = self._display_offset(viz_vae,
                                              offset=[0.0, offsets_y[i], 1.4])
            self.raw_q0_vae.append(copy.deepcopy(raw_q0_vae))
            self.raw_viz_vae.append(viz_vae)
            del viz_vae

    def _display_offset(self, viz, offset):
        q = self.q0.copy()
        q[0] += offset[0]
        q[1] += offset[1]
        q[2] += offset[2]
        viz.display(q)
        return q

    def _get_slider_values(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect((self.server_host, self.server_port))
                sock.sendall(b"get_values\n")
                data = sock.recv(10240)
                return json.loads(data.decode())
        except Exception as e:
            print("[Socket error]", e)
            return {
                "raw": [0.0] * self.num_joints,
                "vae": [],
                "pca": [],
                "control": [0.0] * self.num_joints
            }

    def extract_finger_joints(self, joints):

        if np.array(joints).shape[0] > self.num_joints:

            return joints[:self.num_joints]

        raw_joints = (np.array(joints) + 1) / 2 * (
            self.joint_limits[:, 1] -
            self.joint_limits[:, 0]) + self.joint_limits[:, 0]
        return raw_joints.tolist()

    def run(self, frequency_hz=30):
        dt = 1.0 / frequency_hz
        while True:
            values = self._get_slider_values()

            # === RAW update ===
            raw = values.get("raw") or [0.0] * self.num_joints
            q_raw = self.q0.copy()

            q_raw[-len(raw):] = self.extract_finger_joints(raw)

            self.viz_main.display(q_raw)

            # === demo update ===
            demo_data = values.get("demo") or [0.0] * self.num_joints
            q_demo = self.q0_demo.copy()
            q_demo[-len(demo_data):] = self.extract_finger_joints(demo_data)
            self.viz_demo.display(q_demo)

            # === PCA update ===
            pca_data = values.get("pca") or [0.0] * self.num_joints
            q_pca = self.q0_pca.copy()

            q_pca[-len(pca_data):] = self.extract_finger_joints(pca_data)
            self.viz_pca.display(q_pca)

            # === Optional: VAE (static or not used) ===
            # You can also add q_vae update here if needed.
            vae_data = values.get("vae") or [[0.0] * self.num_joints]

            for index, vae_action in enumerate(vae_data):

                q_vae = self.q0_vae[index].copy()
                q_vae[-self.num_joints:] = self.extract_finger_joints(
                    vae_action)
                self.viz_vae[index].display(q_vae)

            reconstructed_vae_data = values.get("vae_reconstructed") or [
                [0.0] * self.num_joints * 2
            ]

            for index, vae_action in enumerate(reconstructed_vae_data):

                num_finger_actions = len(vae_action) // 2

                reconstructed_q_vae = self.reconstructed_q0_vae[index].copy()

                reconstructed_q_vae[-self.
                                    num_joints:] = self.extract_finger_joints(
                                        vae_action[:num_finger_actions])
                self.reconstructed_viz_vae[index].display(reconstructed_q_vae)

                raw_q_vae = self.raw_q0_vae[index].copy()

                raw_q_vae[-self.num_joints:] = self.extract_finger_joints(
                    vae_action[num_finger_actions:])
                self.raw_viz_vae[index].display(raw_q_vae)

            time.sleep(dt)


if __name__ == "__main__":
    model_path = "source/assets/robot/leap_hand_v2/archive/leap_hand"
    urdf_filename = "leap_hand_right_glb.urdf"
    urdf_path = join(model_path, urdf_filename)

    visualizer = HandVisualizerClient(urdf_path=urdf_path,
                                      model_path=model_path,
                                      server_host="localhost",
                                      num_vae=2,
                                      server_port=10112)
    visualizer.run()
