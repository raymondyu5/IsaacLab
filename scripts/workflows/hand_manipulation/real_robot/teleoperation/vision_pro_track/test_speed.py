import threading
import grpc
import time
import torch
import numpy as np
from scripts.workflows.hand_manipulation.real_robot.teleoperation.vision_pro_track import handtracking_pb2, handtracking_pb2_grpc

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Matrix Processing ---
def process_matrix_torch(message):
    return torch.tensor(
        [[message.m00, message.m01, message.m02, message.m03],
         [message.m10, message.m11, message.m12, message.m13],
         [message.m20, message.m21, message.m22, message.m23], [0, 0, 0, 1]],
        dtype=torch.float32,
        device=DEVICE).unsqueeze(0)


def process_matrices_torch(skeleton, matrix=None):
    num_joints = len(skeleton)
    joint_array = torch.zeros((num_joints, 4, 4),
                              dtype=torch.float32,
                              device=DEVICE)

    for i, joint in enumerate(skeleton):
        joint_array[i, :3, 0] = torch.tensor([joint.m00, joint.m01, joint.m02],
                                             device=DEVICE)
        joint_array[i, :3, 1] = torch.tensor([joint.m10, joint.m11, joint.m12],
                                             device=DEVICE)
        joint_array[i, :3, 2] = torch.tensor([joint.m20, joint.m21, joint.m22],
                                             device=DEVICE)
        joint_array[i, :3, 3] = torch.tensor([joint.m03, joint.m13, joint.m23],
                                             device=DEVICE)
        joint_array[i, 3, 3] = 1.0

    if matrix is None:
        matrix = torch.eye(4, dtype=torch.float32, device=DEVICE)

    return torch.matmul(matrix.unsqueeze(0),
                        joint_array)  # (1, 4, 4) x (N, 4, 4) => (N, 4, 4)


def expand_to_homogeneous(R: torch.Tensor) -> torch.Tensor:
    """
    Converts a (3, 3) rotation matrix to a (4, 4) homogeneous transform.
    """
    T = torch.eye(4, device=R.device, dtype=R.dtype)
    T[:3, :3] = R
    return T


def two_mat_batch_mul_torch(A, B):
    if B.dim() == 2:
        B = B.unsqueeze(0)
    return torch.matmul(A, B.expand(A.shape[0], -1, -1))


def joint_avp2hand_torch(finger_mat):
    finger_index = torch.tensor([
        0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23,
        24
    ],
                                device=DEVICE)
    return finger_mat[finger_index]


def extract_finger_positions(joint_pose):
    return joint_pose[:, :3, 3]  # (N, 3)


# --- Transforms ---
OPERATOR2AVP_LEFT = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]],
                                 dtype=torch.float32,
                                 device=DEVICE)
OPERATOR2AVP_RIGHT = torch.tensor([[0, 0, -1], [-1, 0, 0], [0, 1, 0]],
                                  dtype=torch.float32,
                                  device=DEVICE)


def three_mat_mul(left_rot: torch.Tensor, mat: torch.Tensor,
                  right_rot: torch.Tensor):

    result = torch.eye(4, device=mat.device, dtype=mat.dtype)
    rotation = left_rot @ mat[:3, :3] @ right_rot
    translation = left_rot @ mat[:3, 3]
    result[:3, :3] = rotation
    result[:3, 3] = translation

    return result


def rotate_head_torch(R: torch.Tensor, degrees: float = -90) -> torch.Tensor:
    theta = torch.deg2rad(torch.tensor(degrees, device=R.device,
                                       dtype=R.dtype))

    R_x = torch.tensor(
        [[1, 0, 0, 0], [0, torch.cos(theta), -torch.sin(theta), 0],
         [0, torch.sin(theta), torch.cos(theta), 0], [0, 0, 0, 1]],
        device=R.device,
        dtype=R.dtype)

    return R @ R_x


# --- Teleoperation Server ---
class TeleopServer:

    def __init__(self, avp_address="192.168.0.50", port=8888):
        self.avp_address = avp_address
        self.port = port
        self.latest_transformation = {}
        self.axis_transform = torch.tensor([[0, 0, -1], [-1, 0, 0], [0, 1, 0]],
                                           dtype=torch.float32,
                                           device=DEVICE)
        self.latest_command = "Teleoperation is not ready"
        self.failed = True

        # NEW: For async frame handling
        self._latest_response = None
        self._lock = threading.Lock()

    def start_visionpro(self):
        threading.Thread(target=self.stream_from_vision_pro,
                         daemon=True).start()
        threading.Thread(target=self.process_latest_response_loop,
                         daemon=True).start()
        time.sleep(0.01)

    def get_latest_command(self):
        return self.latest_command, self.latest_transformation

    def send_command(self, command):
        send_command_to_vision_pro(self.avp_address, self.port, command)

    def stream_from_vision_pro(self):
        request = handtracking_pb2.HandUpdate()

        while True:
            try:
                print(
                    f"🌐 Attempting to connect to Vision Pro at {self.avp_address}:12345"
                )
                self._frame_counter = 0
                self._last_fps_time = time.time()
                with grpc.insecure_channel(
                        f"{self.avp_address}:12345") as channel:
                    stub = handtracking_pb2_grpc.HandTrackingServiceStub(
                        channel)
                    responses = stub.StreamHandUpdates(request)
                    print("✅ Connected to Vision Pro")

                    for response in responses:
                        # 🔁 Replace queue-style loop with single-frame overwrite
                        with self._lock:
                            self._latest_response = response

                        self._frame_counter += 1
                        # now = time.time()
                        # fps = 1.0 / (now - self._last_fps_time + 1e-8
                        #              )  # small epsilon to prevent div by 0
                        # self._last_fps_time = now
                        # print(f"📸 Vision Pro stream FPS: {fps:.2f}")

                    self.failed = False

            except grpc.RpcError as e:
                print(f"❌ gRPC error: {e}")
                self.failed = True
                self.latest_transformation = {}
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                self.failed = True
                self.latest_transformation = {}

            print("🔁 Reconnecting to Vision Pro in 3 seconds...")
            time.sleep(3 if self.failed else 0.01)

    def process_latest_response_loop(self):
        while True:
            response = None
            with self._lock:
                if self._latest_response is not None:
                    response = self._latest_response
                    self._latest_response = None  # consume it

            if response is not None:
                try:
                    # Process frame
                    left_joints = process_matrices_torch(
                        response.left_hand.skeleton.jointMatrices)
                    right_joints = process_matrices_torch(
                        response.right_hand.skeleton.jointMatrices)

                    left_transform = expand_to_homogeneous(OPERATOR2AVP_LEFT.T)
                    right_transform = expand_to_homogeneous(
                        OPERATOR2AVP_RIGHT.T)

                    left_joints = two_mat_batch_mul_torch(
                        left_joints, left_transform)
                    right_joints = two_mat_batch_mul_torch(
                        right_joints, right_transform)

                    left_joint_pose = joint_avp2hand_torch(left_joints)
                    right_joint_pose = joint_avp2hand_torch(right_joints)

                    left_fingers = extract_finger_positions(
                        left_joint_pose).detach().cpu().numpy()
                    right_fingers = extract_finger_positions(
                        right_joint_pose).detach().cpu().numpy()

                    self.latest_transformation = {
                        "left_fingers":
                        left_fingers,
                        "right_fingers":
                        right_fingers,
                        "left_wrist":
                        three_mat_mul(
                            self.axis_transform,
                            process_matrix_torch(
                                response.left_hand.wristMatrix)[0],
                            OPERATOR2AVP_LEFT,
                        ).detach().cpu().numpy(),
                        "right_wrist":
                        three_mat_mul(
                            self.axis_transform,
                            process_matrix_torch(
                                response.right_hand.wristMatrix)[0],
                            OPERATOR2AVP_RIGHT,
                        ).detach().cpu().numpy(),
                    }

                    self.latest_command = response.command

                except Exception as e:
                    print(f"❌ Error processing frame: {e}")

            time.sleep(0.01)


# --- Main ---
if __name__ == "__main__":
    server = TeleopServer("10.0.0.160")
    server.start_visionpro()

    while True:
        latest_command, pose = server.get_latest_command()
        print(latest_command)
        time.sleep(0.01)
