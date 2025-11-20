import numpy as np
import pickle
from pathlib import Path
from typing import List

OPERATOR2MANO_RIGHT = np.array([
    [0, 0, -1],
    [-1, 0, 0],
    [0, 1, 0],
])

OPERATOR2MANO_LEFT = np.array([
    [0, 0, -1],
    [1, 0, 0],
    [0, -1, 0],
])

OPERATOR2AVP_RIGHT = OPERATOR2MANO_RIGHT

OPERATOR2AVP_LEFT = np.array([
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
])

OPERATOR2AVP_CAM = np.array([[1., 0., 0., 0.0], [0., 0., -1., 0.0],
                             [0., 1., 0., 0.0], [0., 0., 0., 1.]])


def three_mat_mul(left_rot: np.ndarray, mat: np.ndarray,
                  right_rot: np.ndarray):
    result = np.eye(4)
    rotation = left_rot @ mat[:3, :3] @ right_rot
    pos = left_rot @ mat[:3, 3]
    result[:3, :3] = rotation
    result[:3, 3] = pos
    return result


def two_mat_batch_mul(batch_mat: np.ndarray, left_rot: np.ndarray):
    result = np.tile(np.eye(4), [batch_mat.shape[0], 1, 1])
    result[:, :3, :3] = np.matmul(left_rot[None, ...], batch_mat[:, :3, :3])
    result[:, :3, 3] = batch_mat[:, :3, 3] @ left_rot.T
    return result


def joint_avp2hand(finger_mat: np.ndarray):
    finger_index = np.array([
        0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23,
        24
    ])
    finger_mat = finger_mat[finger_index]
    return finger_mat


def filter_data(data: List, fps, duration):

    init_time = data[0]["time"]
    all_times = np.array([d["time"] for d in data]) - init_time
    step = 1.0 / fps
    new_data = []
    for i in range(fps * duration):
        current_time = i * step
        diff = np.abs(all_times - current_time)
        best_match = np.argmin(diff)
        new_data.append(data[best_match])
    return new_data
