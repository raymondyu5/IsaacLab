import trimesh
import numpy as np

# Create a unit cube (centered at origin, side length = 1)
cube = trimesh.creation.box(extents=(1.0, 1, 1))
cube_vertices = cube.vertices
cube_vertices[:, 2] -= np.max(cube_vertices[:, 2])
cube = trimesh.Trimesh(vertices=cube_vertices, faces=cube.faces)

# Save as OBJ file
cube.export(
    '/home/ensu/Documents/weird/IsaacLab_assets/assets/table/table_block.obj')
