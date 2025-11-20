import tkinter as tk
from tkinter import ttk
import numpy as np
import socket
import threading
import json
import argparse
import torch
from scripts.workflows.hand_manipulation.utils.meshcatviewer.pca_slider import PCASliderApp
from scripts.workflows.hand_manipulation.utils.meshcatviewer.demo_slider import DemoSliderApp
from scripts.workflows.hand_manipulation.utils.meshcatviewer.vae_slider import VAESliderApp
import math

import h5py


class SliderApp:

    def __init__(self, root, args):
        self.root = root
        self.root.title("Slider Interface")
        self.root.columnconfigure((0, 1), weight=1)
        self.root.rowconfigure((0, 1), weight=1)
        self.device = args.device

        self.pca_path = args.pca_path
        self.hand_side = args.hand_side
        self.demo_path = args.demo_path
        self.vae_path = args.vae_path
        self.init_settings()

    def init_settings(self):
        self.group_count = 0

        self.num_slider = 1

        if self.pca_path is not None:
            self.num_slider += 1

        if self.demo_path is not None:
            self.num_slider += 1
            self.load_demo_data()

        # For VAE, count the number of dimensions (len) if provided
        if self.vae_path is not None:
            self.num_slider += len(self.vae_path)
            self.num_vae = len(self.vae_path)
            if self.demo_path is not None:
                self.num_slider += 1
        else:
            # Add 1 by default if no VAE path is provided
            self.num_slider += 1
            self.num_vae = 0

        # Font styles``
        self.section_font = ("Arial", 14, "bold")
        self.label_font = ("Arial", 12)

        # Define joint names and dimensions
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
        self.num_hand_joints = len(self.raw_joint_names)
        self.vae_dim = 8
        self.raw_sliders = []
        self.vae_sliders = []
        self.pca_sliders = []

        # === Raw Joint Sliders ===
        self._create_slider_group("Raw Joint Angles [-3.14, 3.14]",
                                  self.raw_joint_names, -3.14, 3.14,
                                  self.raw_sliders)

        # === demo Sliders (only if path is given) ===
        if self.demo_path:
            self.demo_app = DemoSliderApp(self.raw_sliders,
                                          root=self.root,
                                          action_buffer=self.action_buffer,
                                          hand_side=self.hand_side,
                                          device=self.device,
                                          group_count=self.group_count,
                                          num_slider=self.num_slider)
            self.group_count = self.demo_app.group_count

        # === VAE Latent Sliders ===
        if self.vae_path is not None:

            self.vae_app = VAESliderApp(
                self.raw_sliders,
                root=self.root,
                action_buffer=self.action_buffer,
                vae_path=self.vae_path,
                hand_side=self.hand_side,
                device=self.device,
                slider_function=self._create_slider_group,
                group_count=self.group_count,
                num_slider=self.num_slider)

            self.group_count = self.vae_app.group_count
        # else:
        #     self._create_slider_group("VAE Latents [-1, 1]",
        #                               [f"z[{i}]" for i in range(self.vae_dim)],
        #                               -2, 2, self.vae_sliders)

        # === PCA Sliders (only if path is given) ===
        if self.pca_path:
            self.pca_app = PCASliderApp(pca_path=self.pca_path,
                                        hand_side=self.hand_side)
            self._create_slider_group(
                f"PCA Latents",
                [f"pca[{i}]" for i in range(len(self.pca_app.min_values))], -1,
                1, self.pca_sliders)

        # Start socket server
        threading.Thread(target=self._start_socket_server, daemon=True).start()

    def load_demo_data(self):

        data = h5py.File(f"{self.demo_path}/raw_{self.hand_side}_data.hdf5",
                         "r")["data"]
        self.action_buffer = []

        for index in range(len(data)):
            # if use_joint_pos:
            #     self.action_buffer.append(
            #         np.array(data[f"demo_{index}"]["obs"]
            #                  [f"{self.hand_side}_hand_joint_pos"])[::1])
            # else:

            self.action_buffer.append(
                np.array(data[f"demo_{index}"]["actions"])[::1])

    def _create_slider_group(self, title, labels, min_val, max_val,
                             slider_list):
        group_frame = ttk.LabelFrame(self.root, padding=10)

        n = self.num_slider  # Total number of groups you want
        cols = math.ceil(math.sqrt(n))

        row = self.group_count // cols
        col = self.group_count % cols
        group_frame.grid(row=row, column=col, sticky="nsew", padx=10, pady=10)
        self.group_count += 1

        title_label = tk.Label(group_frame, text=title, font=self.section_font)
        title_label.pack(anchor="w", pady=(0, 10))

        for i, label_text in enumerate(labels):
            vmin = min_val[i] if isinstance(min_val,
                                            (list, np.ndarray)) else min_val
            vmax = max_val[i] if isinstance(max_val,
                                            (list, np.ndarray)) else max_val
            self._add_slider(group_frame, label_text, float(vmin), float(vmax),
                             slider_list)

        # Buttons
        button_frame = ttk.Frame(group_frame)
        button_frame.pack(anchor="e", pady=(10, 0))

        simulate_btn = ttk.Button(button_frame,
                                  text="Simulate",
                                  command=lambda: self._simulate_gaussian(
                                      slider_list, min_val, max_val))
        simulate_btn.pack(side="left", padx=5)

        reset_btn = ttk.Button(
            button_frame,
            text="Reset",
            command=lambda: self._reset_to_mean(slider_list))
        reset_btn.pack(side="left", padx=5)

    def _add_slider(self, parent, label_text, min_val, max_val, slider_list):
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=8)
        label = ttk.Label(frame,
                          text=label_text,
                          width=10,
                          font=self.label_font)
        label.pack(side="left", padx=(5, 10))

        var = tk.DoubleVar()
        slider = ttk.Scale(frame,
                           from_=min_val,
                           to=max_val,
                           orient="horizontal",
                           variable=var,
                           length=500)
        slider.set(0.0)
        slider.pack(side="left", fill="x", expand=True)
        slider_list.append(var)

    def _simulate_gaussian(self, sliders, min_val, max_val):
        samples = np.random.randn(len(sliders))
        clamped = np.clip(
            samples, min_val,
            max_val) if not isinstance(min_val, (list, np.ndarray)) else [
                np.clip(s, float(min_val[i]), float(max_val[i]))
                for i, s in enumerate(samples)
            ]
        for slider_var, val in zip(sliders, clamped):
            slider_var.set(val)

    def _reset_to_mean(self, sliders):
        for var in sliders:
            var.set(0.0)

    def _get_all_values(self):
        raw = [v.get() for v in self.raw_sliders]

        if self.vae_path is not None:
            # Get VAE values from the VAESliderApp
            vae = self.vae_app.get_vae_values()
        else:
            vae = {"vae": None}
        pca = [v.get() for v in self.pca_sliders]
        demo = [v.get() for v in self.raw_sliders]

        if self.pca_path:
            pca_slider = np.array(pca).reshape(1, -1)  # (1, K)
            reconstructed = self.pca_app.reconstruct_hand_pose_from_normalized_action(
                pca_slider, )

            pca = reconstructed.tolist()  # (D,) as list
        if self.demo_path:
            demo = self.demo_app.get_demo_values().tolist()

        return {"raw": raw, "pca": pca, "demo": demo} | vae

    def _start_socket_server(self, host='localhost', port=10112):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((host, port))
        s.listen(1)
        print(f"[Socket] Listening on {host}:{port}")

        while True:

            conn, addr = s.accept()
            # print(f"[Socket] Connection from {addr}")
            with conn:
                while True:
                    data = conn.recv(20480)
                    if not data:
                        break
                    if data.decode().strip() == "get_values":
                        values = self._get_all_values()
                        conn.sendall(json.dumps(values).encode())
                    else:
                        conn.sendall(b"Unknown command\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pca_path",
                        type=str,
                        default=None,
                        help="Path to PCA .npz.npy file")
    parser.add_argument(
        "--demo_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        nargs='+',  # Accept one or more values
        default=None,
        help="List of VAE path strings")
    parser.add_argument("--hand_side",
                        type=str,
                        default="right",
                        help="left or right hand")
    parser.add_argument("--device",
                        type=str,
                        default="cuda",
                        help="torch device")
    args = parser.parse_args()

    root = tk.Tk()
    app = SliderApp(root, args)
    root.mainloop()
