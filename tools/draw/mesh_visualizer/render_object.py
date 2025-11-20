import pyrender
import trimesh
import os
import numpy as np
# Object directory
object_dir = "source/assets/ycb/dexgrasp/recenter_objects"

# Create an empty scene
scene = pyrender.Scene()

# Load object list
object_files = os.listdir(object_dir)
# remove some unwanted ones
for unwanted in [
        "wood_block", "pitcher_base", "cracker_box", "power_drill", "banana"
]:
    if unwanted in object_files:
        object_files.remove(unwanted)

# Grid config
cols = 4  # number of objects per row
spacing = 0.2  # distance between objects
rows = (len(object_files) + cols - 1) // cols
object_files = ["bleach_cleanser"]

for i, glb_file in enumerate(object_files):
    obj_file = os.path.join(object_dir, glb_file, "textured_recentered.glb")
    mesh_trimesh = trimesh.load(obj_file)

    if isinstance(mesh_trimesh, trimesh.Scene):
        mesh_trimesh = trimesh.util.concatenate(
            tuple(mesh_trimesh.geometry.values()))

    mesh = pyrender.Mesh.from_trimesh(mesh_trimesh)

    # Compute row and column
    row = i // cols
    col = i % cols

    # Position in grid (X = column, Y = row)
    x_offset = col * spacing
    y_offset = -row * spacing  # negative to go down
    pose = [[1, 0, 0, x_offset], [0, 1, 0, y_offset], [0, 0, 1, 0],
            [0, 0, 0, 1]]

    scene.add(mesh, pose=pose)

# Add a camera looking at the grid
camera = pyrender.PerspectiveCamera(yfov=3.14159 / 3.0)
cam_pose = [[1, 0, 0, 0.5], [0, 1, 0, -0.5], [0, 0, 1, 1.0], [0, 0, 0, 1]]
scene.add(camera, pose=cam_pose)

# Add light
light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
scene.add(light, pose=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 2], [0, 0, 0, 1]])

# Viewer
pyrender.Viewer(scene, use_raymond_lighting=True)
