import trimesh
import numpy as np

# Define the dimensions of the cube (length, width, height)
dimensions = [100.0, 100.0, 100.0]  # This creates a unit cube

# Create the cube
cube = trimesh.creation.box(extents=dimensions)

# Calculate the translation vector to move the origin to the top face
translation = [0, 0, dimensions[2] / 2]

# Apply the translation
cube.apply_translation(translation)
cube.export("cube.obj")

# Visualize the cube
cube.show()
