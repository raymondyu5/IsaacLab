import numpy as np
from plyfile import PlyData, PlyElement
# import open3d as o3d
from tools.visualization_utils import *


def SH2RGB(sh):
    """
    Converts from the 0th spherical harmonic coefficient to RGB values [0,1]
    """
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Function to remove outliers using Median Absolute Deviation (MAD)
def remove_far_points(xyz, rgb, threshold_factor=6.0):
    median = np.median(xyz, axis=0)
    diff = np.linalg.norm(xyz - median, axis=1)
    mad = np.median(np.abs(diff - np.median(diff)))
    mask = diff < threshold_factor * mad
    if rgb is None:
        return xyz[mask], None
    return xyz[mask], rgb[mask]


def load_ply(path):

    if isinstance(path, str):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(
            plydata.elements[0]["x"]), np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),
                       axis=1)

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        rgb = SH2RGB(features_dc[..., 0])
        rgb = np.maximum(rgb, 0)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    elif isinstance(path, dict):
        sh = path["pipeline"]['_model.gauss_params.features_dc']
        xyz = path["pipeline"]['_model.gauss_params.means'].cpu().numpy()
        opacities = path["pipeline"]['_model.gauss_params.opacities'].cpu(
        ).numpy()
        rgb = SH2RGB(sh).cpu().numpy()
        rgb = np.maximum(rgb, 0)

    max_rgb = np.max(rgb, axis=1)
    max_rgb = np.maximum(max_rgb, 1)
    rgb = rgb / max_rgb[:, np.newaxis]

    opacities = sigmoid(opacities)
    opacity_mask = (opacities > 0.6).squeeze(1)
    xyz = xyz[opacity_mask]
    rgb = rgb[opacity_mask]

    # Filter out points with black RGB color ([0, 0, 0])
    black_color_mask = np.any((rgb * 255).astype(np.uint8) != 0, axis=1)

    xyz = xyz[black_color_mask]
    rgb = rgb[black_color_mask]

    xyz, rgb = remove_far_points(xyz, rgb, threshold_factor=15.0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    return pcd


# import h5py
# with h5py.File(f"logs/rabbit/static_gs/rabbit_0.hdf5", 'r') as file:

#     seg_pc_batch = file["data"]["demo_0"]["obs"]["seg_pc"][0]
#     xyz = seg_pc_batch[:, :, :3]
#     rgb = seg_pc_batch[:, :, 3:6]
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(xyz.reshape(-1, 3))
#     pcd.colors = o3d.utility.Vector3dVector(rgb.reshape(-1, 3) / 255)

# checkpoint = torch.load(
#     '/media/lme/data4/weird/Spring-Gaus/outputs/static_gs/splatfacto/2024-08-29_050250/nerfstudio_models/step-000008000.ckpt'
# )

# # static_pc = load_ply()
# deform_pc_nerfstudio = load_ply(checkpoint)

# o3d.visualization.draw_geometries([deform_pc_nerfstudio, pcd])

# deform_pc = load_ply("/home/lme/Downloads/8_29_2024.ply")
deform_pc_gui = load_ply(
    "logs/rabbit/dataset/checkpoints/gaussian_ply/25000.ply")
# Load the point cloud
pcd = o3d.io.read_point_cloud(
    "/media/lme/data4/weird/IsaacLab/logs/rabbit/metashape/seg_imgs/dataset_0/sparse_pc.ply"
)

xyz = np.asarray(pcd.points)
xyz_max = np.max(xyz, axis=0)  #(,z,)
xyz_min = np.min(xyz, axis=0)
import pdb

pdb.set_trace()
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=1.0, origin=[0, 0, 0])

o3d.visualization.draw_geometries([deform_pc_gui, pcd, coordinate_frame])
