# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Script to an environment with random action agent."""
"""Launch Isaac Sim Simulator first."""

import sys

sys.path.append(".")

from tools.visualization_utils import *

from isaaclab.app import AppLauncher
import cv2
import matplotlib.pyplot as plt
import numpy as np
# import open3d as o3d

import argparse

import time
from tools.visualization_utils import *

from isaaclab.app import AppLauncher
import cv2
import matplotlib.pyplot as plt
import numpy as np
# import open3d as o3d

from isaaclab.utils import Timer
import isaaclab.utils.math as math_utils
from scripts.workflows.utils.parse_setting import save_params_to_yaml, parser
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()


def obtain_target_quat_from_multi_angles(axis, angles):
    quat_list = []
    for index, cam_axis in enumerate(axis):
        euler_xyz = torch.zeros(3)
        euler_xyz[cam_axis] = angles[index]
        quat_list.append(
            math_utils.quat_from_euler_xyz(euler_xyz[0], euler_xyz[1],
                                           euler_xyz[2]))
    if len(quat_list) == 1:
        return quat_list[0]
    else:
        target_quat = quat_list[0]
        for index in range(len(quat_list) - 1):

            target_quat = math_utils.quat_mul(quat_list[index + 1],
                                              target_quat)
        return target_quat


import pdb

pdb.set_trace()
math_utils.euler_xyz_from_quat(
    obtain_target_quat_from_multi_angles([1, 2],
                                         [1.57, 1.57 / 2]).unsqueeze(0))
