# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Keyboard controller for SE(3) control."""

import numpy as np
import weakref
from collections.abc import Callable
from scipy.spatial.transform import Rotation

import carb
import omni
from source.isaaclab.isaaclab.devices import DeviceBase

from scripts.workflows.hand_manipulation.utils.keyboard_interface
class Se3Keyboard(DeviceBase):

    def __init__(self, ):
        """Initialize the keyboard layer.

        Args:
            pos_sensitivity: Magnitude of input position command scaling. Defaults to 0.05.
            rot_sensitivity: Magnitude of scale input rotation commands scaling. Defaults to 0.5.
        """
        # store inputs

        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        # note: Use weakref on callbacks to ensure that this object can be deleted when its destructor is called.
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj.
            _on_keyboard_event(event, *args),
        )
        self._create_key_bindings()
        self.start_alignment, self.exit_camera_adjustment = False, False

        # dictionary for additional callbacks
        self._additional_callbacks = dict()

    def __del__(self):
        """Release the keyboard interface."""
        self._input.unsubscribe_from_keyboard_events(self._keyboard,
                                                     self._keyboard_sub)
        self._keyboard_sub = None

    def __str__(self) -> str:
        """Returns: A string containing the information of joystick."""
        msg = f"Keyboard Controller for SE(3): {self.__class__.__name__}\n"
        msg += f"\tKeyboard name: {self._input.get_keyboard_name(self._keyboard)}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tBegin vision pro alignment: V\n"
        msg += "\tExit Camera Adjustment: C\n"

        return msg

    """
    Operations
    """

    def reset(self):
        # default flags
        pass

    def add_callback(self, key: str, func: Callable):
        """Add additional functions to bind keyboard.

        A list of available keys are present in the
        `carb documentation <https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html>`__.

        Args:
            key: The keyboard button to check against.
            func: The function to call when key is pressed. The callback function should not
                take any arguments.
        """
        self._additional_callbacks[key] = func

    def advance(self):
        """Provides the result from keyboard event state.

        Returns:
            A tuple containing the delta pose command and gripper commands.
        """
        # convert to rotation vector
        return self.start_alignment, self.exit_camera_adjustment

    """
    Internal helpers.
    """

    def _on_keyboard_event(self, event, *args, **kwargs):
        """Subscriber callback to when kit is updated.

        Reference:
            https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html
        """
        # apply the command when pressed

        self.start_alignment = False
        self.exit_camera_adjustment = False

        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "V":
                self.start_alignment = True
            if event.input.name == "C":
                self.exit_camera_adjustment = True
        # additional callbacks
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._additional_callbacks:
                self._additional_callbacks[event.input.name]()

        # since no error, we are fine :)
        return True

    def _create_key_bindings(self):
        """Creates default key binding."""
        self._INPUT_KEY_MAPPING = {
            # toggle: gripper command
            "V": True,
            "C": True,
        }
