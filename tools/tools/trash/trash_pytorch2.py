import torch
import numpy as np
from skimage import measure  # For Marching Cubes
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    PerspectiveCameras,
    PointLights,
    look_at_view_transform,
)
import matplotlib.pyplot as plt
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
# Path to the .obj file
obj_filename = "/home/ensu/Downloads/mesh-simplify.obj"

# Load obj file
mesh = load_objs_as_meshes([obj_filename], device=device)

# Camera setup
R, T = look_at_view_transform(2.7, 0, 0)  # Distance, elevation, azimuth
cameras = PerspectiveCameras(device=device, R=R, T=T)

# Lighting setup
lights = PointLights(device=device, location=[[2.0, 2.0, -2.0]])

# Rasterization settings
raster_settings = RasterizationSettings(
    image_size=512,
    blur_radius=0.0,
    faces_per_pixel=1,
)

# Renderer setup
renderer = MeshRenderer(rasterizer=MeshRasterizer(
    cameras=cameras, raster_settings=raster_settings),
                        shader=HardPhongShader(device=device,
                                               cameras=cameras,
                                               lights=lights))

# Render the mesh
images = renderer(mesh)

# Visualize the rendered image
plt.imshow(images[0, ..., :3].cpu().numpy())
plt.axis("off")
plt.show()
