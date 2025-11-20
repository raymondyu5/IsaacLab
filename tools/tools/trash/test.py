from scipy.spatial.transform import Rotation as R

# Assuming you have a rotation matrix `rot_matrix` (3x3)
rot_matrix = [[1.0, 0., -00], [-0., -1, 0.0], [0.0, 0.0, 1.0]]

# Create a Rotation object from the rotation matrix
r = R.from_matrix(rot_matrix)

# Convert to Euler angles
# You can specify the order of rotations, e.g., 'xyz', 'zyx', etc.
euler_angles = r.as_euler(
    'xyz', degrees=True
)  # 'xyz' is the sequence of rotations; set degrees=False for radians

print("Euler angles (degrees):", euler_angles)
