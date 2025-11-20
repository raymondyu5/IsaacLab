import os
import torch
import matplotlib.pyplot as plt

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import TexturesVertex
# Data structures and functions for rendering
from pytorch3d.renderer import (look_at_view_transform, FoVPerspectiveCameras,
                                PointLights, RasterizationSettings,
                                MeshRenderer, MeshRasterizer, SoftPhongShader,
                                HardPhongShader)
from pytorch3d.structures import Meshes  # Import Meshes
# Setup device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

import isaaclab.utils.math as math_utils


def obtain_target_quat_from_multi_angles(axis, angles):
    quat_list = []
    for index, cam_axis in enumerate(axis):
        euler_xyz = torch.zeros(3)

        euler_xyz[cam_axis] = angles[index]
        quat_list.append(
            math_utils.quat_from_euler_xyz(euler_xyz[0], euler_xyz[1],
                                           euler_xyz[2]))
    if len(quat_list) == 1:
        return torch.as_tensor(quat_list[0], dtype=torch.float16)
    else:
        target_quat = quat_list[0]
        for index in range(len(quat_list) - 1):

            target_quat = math_utils.quat_mul(quat_list[index + 1],
                                              target_quat)
        return torch.as_tensor(target_quat, dtype=torch.float16)


# Path to the .obj file
obj_filename = "/home/ensu/Downloads/texture_mesh/mesh-simplify.obj"

# Load obj file
mesh = load_objs_as_meshes([obj_filename], device=device)

import numpy as np

vertices = mesh.verts_packed()
vertices -= vertices.mean(dim=0)
vertices[:, 2] -= 0.2
vertices[:, 1] -= 1.0
rotation_matrix = torch.as_tensor(math_utils.matrix_from_quat(
    obtain_target_quat_from_multi_angles([0],
                                         [-np.pi / 2]).to(device))[:3, :3],
                                  dtype=torch.float32)

# Apply the rotation matrix to the vertices of the mesh
rotated_verts = torch.matmul(vertices, rotation_matrix.T)

# rotated_verts -= rotated_verts.mean(dim=0)
# rotated_verts[:, 0] += 0.1
num_vertices = mesh.verts_packed().shape[0]
vertex_colors = torch.zeros((1, num_vertices, 3),
                            device=device)  # Shape: (1, V, 3)

# Set the red channel to 1.0 for all vertices
vertex_colors[..., 0] = 1.0
# Create a new mesh with the rotated vertices
mesh = Meshes(verts=[rotated_verts],
              faces=mesh.faces_list(),
              textures=mesh.textures)

# from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
# # Render the plotly figure
# fig = plot_scene({"subplot1": {"cow_mesh": mesh}})
# fig.show()
# import pdb

# pdb.set_trace()

# Initialize a camera
R, T = look_at_view_transform(0.7, 0, 0)
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

# Define rasterization and shading settings
raster_settings = RasterizationSettings(
    image_size=512,
    blur_radius=0.0,
    faces_per_pixel=1,
)

# Place a point light in front of the object
lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

# Create a Phong renderer
renderer = MeshRenderer(rasterizer=MeshRasterizer(
    cameras=cameras, raster_settings=raster_settings),
                        shader=SoftPhongShader(device=device,
                                               cameras=cameras,
                                               lights=lights))
# renderer = MeshRenderer(rasterizer=MeshRasterizer(
#     cameras=cameras, raster_settings=raster_settings),
#                         shader=HardPhongShader(device=device,
#                                                cameras=cameras,
#                                                lights=lights))

# Render the rotated mesh
batch_size = 2
meshes = mesh.extend(batch_size)

# Get a batch of viewing angles
elev = torch.linspace(0, 180, batch_size) * 0.0 + 10
azim = torch.linspace(-180, 180, batch_size) * 0.0

R, T = look_at_view_transform(dist=1.0, elev=elev, azim=azim)
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

images = renderer(meshes, cameras=cameras, lights=lights)

# Visualize the results
from plot_image_grid import image_grid

image_grid(images.cpu().numpy(), rows=1, cols=2, rgb=True)
plt.savefig("image_grid.png")
import cv2
# Load the image (replace 'path_to_image' with your image path)
image = cv2.imread("image_grid.png")  # OpenCV loads images in BGR format

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Display the grayscale image using Matplotlib
plt.imshow(gray_image, cmap="gray")
plt.axis("off")
plt.show()
