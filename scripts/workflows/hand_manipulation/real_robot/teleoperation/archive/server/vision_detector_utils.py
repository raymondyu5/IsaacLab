import numpy as np

HAND_CONNECTIONS = [
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16),
    (0, 17, 18, 19, 20),
]

MANO2ROBOT = np.array([
    [-1, 0, 0],
    [0, 0, -1],
    [0, -1, 0],
])

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


def three_mat_mul(left_rot: np.ndarray, mat: np.ndarray,
                  right_rot: np.ndarray):
    result = np.eye(4)
    rotation = left_rot @ mat[:3, :3] @ right_rot
    pos = left_rot @ mat[:3, 3]
    result[:3, :3] = rotation
    result[:3, 3] = pos
    return result


def rotate_head(R, degrees=-90):
    # Convert degrees to radians
    theta = np.radians(degrees)
    # Create the rotation matrix for rotating around the x-axis
    R_x = np.array([
        [1, 0, 0, 0],
        [0, np.cos(theta), -np.sin(theta), 0],
        [0, np.sin(theta), np.cos(theta), 0],
        [0, 0, 0, 1],
    ])
    R_rotated = R @ R_x
    return R_rotated


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
