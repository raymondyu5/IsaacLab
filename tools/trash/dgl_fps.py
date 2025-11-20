import torch
import dgl


def farthest_point_sampling(points, num_samples):
    """
    Farthest point sampling implemented in PyTorch.

    Args:
        points (torch.Tensor): Input point cloud of shape (N, D), where N is the number of points and D is the dimension (e.g., 3 for 3D points).
        num_samples (int): The number of samples to select.

    Returns:
        torch.Tensor: Indices of the sampled points.
    """
    N, D = points.shape
    sampled_indices = torch.zeros(num_samples,
                                  dtype=torch.long,
                                  device=points.device)
    distances = torch.full((N, ), float('inf'), device=points.device)

    # Start with a random initial point
    sampled_indices[0] = torch.randint(0, N, (1, ), device=points.device)
    farthest_point = points[sampled_indices[0]]

    for i in range(1, num_samples):
        # Calculate the squared Euclidean distance from the farthest point to all other points
        dist = torch.sum((points - farthest_point)**2, dim=1)
        distances = torch.minimum(distances, dist)

        # Select the next farthest point based on maximum distance
        sampled_indices[i] = torch.argmax(distances)
        farthest_point = points[sampled_indices[i]]

    return sampled_indices


# Example usage with DGL
# Generate a random point cloud with 1000 points in 3D, and place it on the GPU
points = torch.rand(100000, 3, device='cuda')
num_samples = 10

# Perform farthest point sampling
sampled_indices = farthest_point_sampling(points, num_samples)

# Retrieve the sampled points
sampled_points = points[sampled_indices]
print("Sampled Points:\n", sampled_points)
