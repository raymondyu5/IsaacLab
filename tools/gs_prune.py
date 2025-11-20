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


def k_nearest_sklearn(x, k: int):
    """
        Find k-nearest neighbors using sklearn's NearestNeighbors.
    x: The data tensor of shape [num_samples, num_features]
    k: The number of neighbors to retrieve
    """
    # Convert tensor to numpy array
    x_np = x

    # Build the nearest neighbors model
    from sklearn.neighbors import NearestNeighbors

    nn_model = NearestNeighbors(n_neighbors=k + 1,
                                algorithm="auto",
                                metric="euclidean").fit(x_np)

    # Find the k-nearest neighbors
    distances, indices = nn_model.kneighbors(x_np)

    # Exclude the point itself from the result and return
    return distances[:, 1:].astype(np.float32), indices[:,
                                                        1:].astype(np.float32)


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
    opacity_mask = (opacities > 0.3).squeeze(1)
    xyz = xyz[opacity_mask]
    rgb = rgb[opacity_mask]

    # Filter out points with black RGB color ([0, 0, 0])
    black_color_mask = np.any((rgb * 255).astype(np.uint8) >= 10, axis=1)

    xyz = xyz[black_color_mask]
    rgb = rgb[black_color_mask]

    xyz, rgb = remove_far_points(xyz, rgb, threshold_factor=15.0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    return pcd


checkpoint = torch.load(
    '/media/lme/data4/weird/IsaacLab/logs/rabbit/dataset/checkpoints/gaussian_ply/20250.ply'
)

# static_pc = load_ply()
deform_pc_nerfstudio = load_ply(checkpoint)

o3d.visualization.draw_geometries([deform_pc_nerfstudio])
