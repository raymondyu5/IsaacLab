import torch
import kaolin.metrics.pointcloud as pc

# Create two random point clouds with shape [batch_size, num_points, 3]
batch_size = 2
num_points = 1000
point_cloud1 = torch.rand((batch_size, num_points, 3), device='cuda')
point_cloud2 = torch.rand((batch_size, num_points, 3), device='cuda')

# Compute the Chamfer distance
chamfer_dist = pc.chamfer_distance(point_cloud1, point_cloud2)

print(f"Chamfer Distance: {chamfer_dist}")
