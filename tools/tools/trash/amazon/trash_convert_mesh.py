import trimesh
import numpy as np
from scipy.spatial.transform import Rotation as R

mesh = trimesh.load("/home/ensu/Downloads/palm_upper_with_holes.stl")

mesh.apply_scale([0.001, 0.001, 0.001])

# euler_angles_rad = np.radians([0, -90, -180])  # Convert degrees to radians

# # Specify the order of rotations, e.g., 'xyz' means rotate around x, then y, then z
# rotation = R.from_euler('xyz', euler_angles_rad)

# # Get the 3x3 rotation matrix
# rotation_matrix = rotation.as_matrix()
# trasnformation_matrix = np.eye(4)
# trasnformation_matrix[:3, :3] = rotation_matrix
# trasnformation_matrix[:3, 3] = [-0.0375, 0.012, 0.045]

# mesh.apply_transform(trasnformation_matrix)

mesh.export("/home/ensu/Downloads/palm_upper.stl")
