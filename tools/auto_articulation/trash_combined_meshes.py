import os
# import open3d as o3d
from omni.isaac.kit import SimulationApp
import numpy as np

simulation_app = SimulationApp({"headless": True})
from pxr import Usd, UsdGeom, Vt
from pxr import UsdPhysics, UsdUtils, Sdf, PhysxSchema


def create_rigid_collision(prim, set_sdf=True, sdf_resolution=512):
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.CollisionAPI.Apply(prim)
    meshcollisionAPI = UsdPhysics.MeshCollisionAPI.Apply(prim)

    meshcollisionAPI.CreateApproximationAttr().Set("sdf")
    meshCollision = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)
    meshCollision.CreateSdfResolutionAttr().Set(sdf_resolution)
    # else:
    #     meshcollisionAPI.CreateApproximationAttr().Set("convexDecomposition")


# Folder containing .ply files
ply_folder = "/home/ensu/Downloads/digital_cousin/meshes/uw_kitchen_4/scene_0"
usd_output = "combined_mesh.usd"

# Create a new USD stage
stage = Usd.Stage.CreateNew(usd_output)

# Root layer for all meshes
root = UsdGeom.Xform.Define(stage, "/Root")

# Iterate through all .ply files in the folder
# import pdb

# pdb.set_trace()
# for idx, ply_file in enumerate(os.listdir(ply_folder)):
#     if ply_file.endswith(".ply"):
for i in range(24, 29):
    mesh_id = f"mesh_{i:04d}"
    ply_file = f"{mesh_id}.ply"
    # Read .ply file
    file_path = os.path.join(ply_folder, ply_file)
    mesh = o3d.io.read_triangle_mesh(file_path)

    # Get vertex and face data
    vertices = mesh.vertices
    faces = mesh.triangles

    # Create a USD mesh under the root
    mesh_name = f"{mesh_id}"
    xform = UsdGeom.Xform.Define(stage, f"/Root/{mesh_id}")
    usd_mesh = UsdGeom.Mesh.Define(stage, f"/Root/{mesh_id}/{mesh_name}")

    points = np.asarray(mesh.vertices).reshape(-1, 3)
    faces = np.asanyarray(mesh.triangles).reshape(-1, 3)
    vertex_counts = np.ones_like(faces[:, 0]).reshape(-1).astype(np.int32) * 3
    usd_mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    usd_mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces))
    usd_mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(vertex_counts))

    prim = stage.GetPrimAtPath(f"/Root/{mesh_id}/{mesh_name}")

    create_rigid_collision(prim, set_sdf=True)

    # Set vertex positions

    print(f"Added {ply_file} as {mesh_name} to USD stage.")

# Save the stage
stage.Save()
print(f"USD file saved to {usd_output}")
