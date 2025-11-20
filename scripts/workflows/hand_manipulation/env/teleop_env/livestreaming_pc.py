import viser
import numpy as np
import time
import torch
from scipy.spatial.transform import Rotation as R
import isaaclab.utils.math as math_utils

TRANSFORMATION_MATRIX = [
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.5, 0.0, 0.707, 0.0, 0.0, 0.707],
    [-0.0, -0.5, -0.0, 0.707, 0.0, 0.0, -0.707],
]
# import pdb

# pdb.set_trace()
# delta_quat01 = torch.as_tensor([[0.707, 0.0, 0.0, -0.707]])
# delta_quat02 = torch.as_tensor([[0.707, 0.0, 0.707, 0.0]])
# target_quat = math_utils.quat_mul(delta_quat02, delta_quat01)
# quat = math_utils.quat_from_euler_xyz(torch.as_tensor([0.0, 0.0, 0.0]))

# TRANSFORMATION_MATRIX = [
#     [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
#     [0.0, 0.5, 0.0, 0.4998, 0.4998, 0.4998, 0.4998],
#     [-0.0, -0.5, -0.0, 0.4998, -0.4998, 0.4998, -0.4998],
# ]

# TRANSFORMATION_MATRIX = [
#     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     [0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
#     [-0.0, -0.5, -0.0, 0.0, 0.0, 0.0],
# ]


class LiveStreamPCViewer:

    def __init__(self, env, host='0.0.0.0', port=8080, num_views=3):
        self.server = viser.ViserServer(host=host, port=port)
        self.env = env
        self.device = env.unwrapped.device
        obs, info = self.env.reset()

        dim = obs["policy"]["robot_synthesize_pc"].cpu().numpy().shape
        self.dim = dim[1]
        self.num_views = num_views
        self.transformation_pose = torch.as_tensor(TRANSFORMATION_MATRIX).to(
            self.device)[:, :3]

        # euler_xyz = torch.as_tensor(TRANSFORMATION_MATRIX)[:, 3:]
        # self.transformation_matrix = math_utils.quat_from_euler_xyz(
        #     euler_xyz[:, 0], euler_xyz[:, 1], euler_xyz[:, 2]).to(self.device)
        self.transformation_matrix = torch.as_tensor(TRANSFORMATION_MATRIX).to(
            self.device)[:, 3:]
        self.colors = obs["policy"]["robot_synthesize_pc"][0, :,
                                                           3:].cpu().numpy()

        self.cloud = self.server.scene.add_point_cloud(
            "live_cloud",
            points=np.zeros((dim[1] * num_views, 3), dtype=np.float32),
            colors=np.zeros((dim[1] * num_views, 3), dtype=np.float32),
            point_size=0.005)

        self.server.scene.world_axes.visible = True
        self.count = 0

    def run(self):
        while True:
            actions = torch.rand(self.env.action_space.shape,
                                 device=self.env.unwrapped.device) * 0.0
            obs, reward, terminate, time_out, info = self.env.step(actions)

            points = obs["policy"]["robot_synthesize_pc"].repeat_interleave(
                self.num_views, dim=0)
            transformed_points = math_utils.transform_points(
                points[..., :3].clone(), None, self.transformation_matrix)
            offset = self.transformation_pose.unsqueeze(1).repeat_interleave(
                transformed_points.shape[1], dim=1)

            # transformed_points = math_utils.transform_points(
            #     transformed_points, None,
            #     torch.as_tensor([[
            #         0.9238795325112867, 0.0, -0.3826834323650898, 0
            #     ]]).to(self.device).repeat_interleave(self.num_views, 0))

            transformed_points = transformed_points + offset
            # transformed_points = transformed_points.reshape(1, -1, 3)

            self.cloud.points = transformed_points.cpu().numpy().reshape(-1, 3)
            self.cloud.colors = points[..., 3:].reshape(-1, 3).cpu().numpy()

            if self.count % 20 == 0:

                # Set camera pose for all connected clients
                for client in list(self.server.get_clients().values()):

                    postion = self.env.scene["wood_block"]._data.root_state_w[
                        0, :3].cpu().numpy()

                    client.camera.look_at = (postion[0], 0.0, 0.0)
                    client.camera.position = (2, 0.0, 2.0)

            self.count += 1
