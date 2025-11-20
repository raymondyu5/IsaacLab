import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import yaml
import numpy as np

# File path to the YAML file
file_path = "/home/ensu/Documents/weird/curobo/src/curobo/content/configs/robot/spheres/franka.yml"

# Load YAML data
with open(file_path, "r") as file:
    collision_data = yaml.safe_load(file)


# Function to plot a sphere
def plot_sphere(ax, center, radius, color='blue'):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=0.5)


# Extract collision spheres for the robot 'wx250s'
collision_spheres = collision_data['collision_spheres']

# Iterate through links and visualize all spheres for each link in one plot
for link, spheres in collision_spheres.items():
    # Calculate the center of the mesh (average of sphere centers)
    centers = np.array([sphere["center"] for sphere in spheres])
    mesh_center = np.mean(centers, axis=0)

    # Initialize 3D plot for the current link
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot all spheres for the current link
    for sphere in spheres:
        center = sphere["center"]
        radius = sphere["radius"]
        plot_sphere(ax, center, radius)

# Plot the mesh center
ax.scatter(*mesh_center, color='red', s=100, label="Mesh Center")
ax.text(mesh_center[0],
        mesh_center[1],
        mesh_center[2],
        f"({mesh_center[0]:.2f}, {mesh_center[1]:.2f}, {mesh_center[2]:.2f})",
        color='red')
print(link, mesh_center)

# Set axis labels and aspect ratio
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1

# Set title for the current link
ax.set_title(f"Link: {link}")

# Add legend
ax.legend()

# Show the plot and wait for user input before moving to the next link
plt.show()
plt.close(fig)  # Close the figure after displaying
