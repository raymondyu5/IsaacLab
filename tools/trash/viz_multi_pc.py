import h5py
# import open3d as o3d
with h5py.File(f"logs/rabbit/static_gs/rabbit_0.hdf5", 'r') as file:

    seg_pc_batch = file["data"]["demo_0"]["obs"]["seg_pc"][0]
    xyz = seg_pc_batch[:, :, :3]
    rgb = seg_pc_batch[:, :, 3:6]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.reshape(-1, 3))
    pcd.colors = o3d.utility.Vector3dVector(rgb.reshape(-1, 3) / 255)
    o3d.visualization.draw_geometries([pcd])
