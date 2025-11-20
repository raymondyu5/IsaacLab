import trimesh
import isaaclab.utils.math as math_utils
import torch
from scripts.workflows.utils.gs_env.gs_utils.gaussian_model import GaussianModel
from scripts.workflows.utils.gs_env.utils.gs_utils import transform_spherical_harmonics


def transform_rigid_object_gs(xyz, rotation, features_rest, transformation):

    transformed_xyz = math_utils.transform_points(xyz, transformation[:3],
                                                  transformation[3:])

    rotation_matrix = math_utils.matrix_from_quat(rotation)
    new_rotation = math_utils.matrix_from_quat(
        transformation[3:]) @ rotation_matrix
    transformed_rotation = math_utils.quat_from_matrix(new_rotation)
    # transformed_rotation2 = math_utils.quat_mul(transformation[3:].unsqueeze(0).repeat_interleave(rotation.shape[0],dim=0), rotation)

    transformed_features_rest = transform_spherical_harmonics(
        features_rest, math_utils.matrix_from_quat(transformation[3:]))
    return transformed_xyz, transformed_rotation, transformed_features_rest


mesh = trimesh.load(
    "/home/ensu/Documents/weird/IsaacLab_assets/assets/gs_objects/plate/mesh.obj"
)
points = mesh.vertices
faces = mesh.faces
transformed_quaternion = torch.as_tensor([0.7167, .3303, -0.5342, -0.3030])
transformed_vertices = math_utils.transform_points(
    torch.as_tensor(points, dtype=torch.float32), torch.zeros(3),
    transformed_quaternion).numpy()

mean_vertices = transformed_vertices.mean(axis=0)
transformed_vertices -= mean_vertices
transformed_mesh = trimesh.Trimesh(vertices=transformed_vertices, faces=faces)
gs_model = GaussianModel(3)
gs_model.load_ply(
    "/home/ensu/Documents/weird/IsaacLab_assets/assets/gs_objects/plate/splat.ply",
    eval=True)

transformation = torch.cat(
    [torch.as_tensor(mean_vertices), transformed_quaternion])
gs_model._xyz, gs_model._rotation, gs_model._features_rest = transform_rigid_object_gs(
    gs_model._xyz, gs_model._rotation, gs_model._features_rest,
    transformation.to("cuda:0"))
gs_model.save_ply(
    "/home/ensu/Documents/weird/IsaacLab_assets/assets/gs_objects/plate/transformed_splat.ply"
)
transformed_mesh.export(
    "/home/ensu/Documents/weird/IsaacLab_assets/assets/gs_objects/plate/transformed_mesh.obj"
)
