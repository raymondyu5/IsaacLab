import tkinter as tk
from tkinter import ttk
import math


class ButtonWrapper:

    def __init__(
        self,
        root,
        num_demos,
        raw_sliders=None,
    ):

        self.root = root
        self.raw_sliders = raw_sliders
        self.num_demos = num_demos

        self.play_demo = False
        self.stop_demo = False

        self.demo_count = 30
        self.frame_index = 0

    def _demo_play(self):
        self.play_demo = True
        self.stop_demo = False

    def create_demo_control_panel(self,
                                  group_count,
                                  num_slider,
                                  name="Demo Controls"):
        style = ttk.Style()
        style.configure("Large.TLabelframe.Label", font=("Arial", 16, "bold"))

        panel = ttk.LabelFrame(self.root,
                               text=f"{name}",
                               padding=10,
                               style="Large.TLabelframe")

        cols = math.ceil(math.sqrt(num_slider))

        row = group_count // cols
        col = group_count % cols
        panel.grid(row=row, column=col, sticky="nsew", padx=10, pady=10)
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
        return group_count + 1

    def _demo_stop(self):
        self.stop_demo = True
        self.play_demo = False

    def _demo_reset(self):
        self._reset_to_mean(self.raw_sliders)  # reset raw joints
        self.target_index = 0
        self.frame_index = 0

    def _reset_to_mean(self, sliders):
        for var in sliders:
            var.set(0.0)

    def _demo_next(self):
        self.demo_count += 1
        self.target_index = 0
        self.frame_index = 0

    def _demo_prev(self):
        self.demo_count -= 1
        self.demo_count = max(0, self.demo_count)
        self.target_index = 0
        self.frame_index = 0
