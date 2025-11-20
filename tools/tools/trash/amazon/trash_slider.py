import tkinter as tk
from tkinter import ttk
import numpy as np
import socket
import threading
import json
import argparse
import torch


class SliderApp:

    def __init__(self, root, pca_path=None, hand_side="right", device="cpu"):
        self.root = root
        self.root.title("Slider Interface")
        self.root.columnconfigure((0, 1), weight=1)
        self.root.rowconfigure((0, 1), weight=1)
        self.device = device
        self.pca_path = pca_path
        self.hand_side = hand_side

        # Font styles
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

        # === VAE Latent Sliders ===
        self._create_slider_group("VAE Latents [-1, 1]",
                                  [f"z[{i}]" for i in range(self.vae_dim)], -2,
                                  2, self.vae_sliders)

        # === PCA Sliders (only if path is given) ===
        if self.pca_path:
            self.load_pca_data()
            self._create_slider_group(
                f"PCA Latents",
                [f"pca[{i}]" for i in range(len(self.min_values))],
                self.min_values, self.max_values, self.pca_sliders)
        self._create_demo_control_panel()

        # Start socket server
        threading.Thread(target=self._start_socket_server, daemon=True).start()

    def _create_demo_control_panel(self):
        panel = ttk.LabelFrame(self.root, text="Demo Controls", padding=10)
        panel.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)

        button_frame = ttk.Frame(panel)
        button_frame.pack(fill="both", expand=True)

        button_style = ttk.Style()
        button_style.configure("Big.TButton", font=("Arial", 14), padding=10)

        ttk.Button(button_frame,
                   text="Play",
                   style="Big.TButton",
                   command=self._demo_play).pack(fill="x", pady=5)
        ttk.Button(button_frame,
                   text="Stop",
                   style="Big.TButton",
                   command=self._demo_stop).pack(fill="x", pady=5)
        ttk.Button(button_frame,
                   text="Reset",
                   style="Big.TButton",
                   command=self._demo_reset).pack(fill="x", pady=5)
        ttk.Button(button_frame,
                   text="Next",
                   style="Big.TButton",
                   command=self._demo_next).pack(fill="x", pady=5)
        ttk.Button(button_frame,
                   text="Previous",
                   style="Big.TButton",
                   command=self._demo_prev).pack(fill="x", pady=5)

    def _demo_play(self):
        print("Play demo")

    def _demo_stop(self):
        print("Stop demo")

    def _demo_reset(self):
        self._reset_to_mean(self.raw_sliders)  # reset raw joints
        print("Reset demo")

    def _demo_next(self):
        print("Next demo")

    def _demo_prev(self):
        print("Previous demo")

    def reconstruct_hand_pose_from_normalized_action(self, a_scaled,
                                                     eigen_vectors, min_values,
                                                     max_values, D_mean,
                                                     D_std):

        x_norm = np.dot(a_scaled, eigen_vectors)  # (B, D)

        # Step 3: Denormalize
        x = x_norm * D_std + D_mean  # (B, D)

        return x

    def load_pca_data(self):
        print(f"Loading PCA from: {self.pca_path}")
        data = np.load(f"{self.pca_path}/{self.hand_side}_pca.npy",
                       allow_pickle=True).item()

        self.eigen_vectors = np.array(data["eigen_vectors"])  # shape: (K, D)
        self.min_values = np.array(data["lower_values"])  # shape: (K,)
        self.max_values = np.array(data["upper_values"])  # shape: (K,)
        self.D_mean = np.array(data["D_mean"])  # shape: (D,)
        self.D_std = np.array(data["D_std"])  # shape: (D,)

        # Project min and max values through PCA basis
        min_proj = np.dot(self.min_values, self.eigen_vectors)  # (D,)
        max_proj = np.dot(self.max_values, self.eigen_vectors)  # (D,)

        # Compute elementwise min and max, then denormalize
        self.min_orig = np.minimum(min_proj,
                                   max_proj) * self.D_std + self.D_mean
        self.max_orig = np.maximum(min_proj,
                                   max_proj) * self.D_std + self.D_mean

    def _create_slider_group(self, title, labels, min_val, max_val,
                             slider_list):
        group_frame = ttk.LabelFrame(self.root, padding=10)

        # Keep track of how many groups have been added
        if not hasattr(self, 'group_count'):
            self.group_count = 0

        row = self.group_count // 2
        col = self.group_count % 2
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
        vae = [v.get() for v in self.vae_sliders]
        pca = [v.get() for v in self.pca_sliders]

        if self.pca_path:
            a_normalized = np.array(pca).reshape(1, -1)  # (1, K)
            reconstructed = self.reconstruct_hand_pose_from_normalized_action(
                a_normalized,
                self.eigen_vectors,  # shape: (K, D)
                self.min_values,  # (K,)
                self.max_values,  # (K,)
                self.D_mean,  # (D,)
                self.D_std  # (D,)
            )

            pca = reconstructed[0][self.retarget2pin].tolist()  # (D,) as list

        return {"raw": raw, "vae": vae, "pca": pca}

    def _start_socket_server(self, host='localhost', port=9999):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((host, port))
        s.listen(1)
        print(f"[Socket] Listening on {host}:{port}")

        while True:
            conn, addr = s.accept()
            # print(f"[Socket] Connection from {addr}")
            with conn:
                while True:
                    data = conn.recv(1024)
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
    parser.add_argument("--hand_side",
                        type=str,
                        default="right",
                        help="left or right hand")
    parser.add_argument("--device",
                        type=str,
                        default="cpu",
                        help="torch device")
    args = parser.parse_args()

    root = tk.Tk()
    app = SliderApp(root,
                    pca_path=args.pca_path,
                    hand_side=args.hand_side,
                    device=args.device)
    root.mainloop()
