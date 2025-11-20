import tkinter as tk
from tkinter import ttk
import numpy as np

import h5py
from scripts.workflows.hand_manipulation.utils.meshcatviewer.button_wrapper import ButtonWrapper


class DemoSliderApp:

    def __init__(self,
                 raw_sliders,
                 root,
                 action_buffer=None,
                 hand_side="right",
                 device="cpu",
                 group_count=0,
                 num_slider=1):

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

        self.hand_side = hand_side
        self.device = device
        self.raw_sliders = raw_sliders
        self.root = root

        self.demo_count = 0
        self.action_buffer = action_buffer
        self.num_demos = len(self.action_buffer)
        self.group_count = self._create_demo_control_panel(
            group_count, num_slider)

    def _create_demo_control_panel(self, group_count, num_slider):

        self.button_wrapper = ButtonWrapper(self.root,
                                            num_demos=self.num_demos,
                                            raw_sliders=self.raw_sliders)
        group_count = self.button_wrapper.create_demo_control_panel(
            group_count, num_slider)
        return group_count

    def get_demo_values(self):
        if not self.button_wrapper.play_demo:
            return np.array([0] * self.num_hand_joints)

        target_demo_data = self.action_buffer[self.button_wrapper.demo_count %
                                              self.num_demos]
        num_frames = len(target_demo_data)

        target_frame = target_demo_data[self.button_wrapper.frame_index %
                                        num_frames]
        if self.button_wrapper.play_demo:

            self.button_wrapper.frame_index += 1
        return target_frame[-self.num_hand_joints:][self.retarget2pin]
