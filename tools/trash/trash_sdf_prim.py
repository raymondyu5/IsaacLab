from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": True})
from pxr import UsdPhysics, UsdUtils, Sdf, PhysxSchema
from pxr import Usd, UsdGeom

# Load the USD file
stage = Usd.Stage.Open(
    "source/assets/kitchen/usd/cabinet02/raw_data/modified_mobility.usd")

# Print the root layer to verify it loaded correctly
print(f"Root layer: {stage.GetRootLayer().identifier}")


def check_points_recursive(prim):
    """
    Recursively check if a prim or any of its children has a 'points' attribute.
    """
    # Check if the current prim has the 'points' attribute
    if prim.GetAttribute("points") and prim.GetAttribute(
            "points").Get() is not None:
        target_prim = stage.GetPrimAtPath(prim.GetPath())
        # if "collision" in str(target_prim.GetPath()) and "handle" in str(
        #         target_prim.GetPath()):
        # Found the 'points' attribute in this prim
        create_rigid_collision(target_prim)

        return True

    # Recursively check all children
    for child in prim.GetChildren():
        if check_points_recursive(child):
            return True

    return False


def create_rigid_collision(prim, sdf_resolution=512):
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.CollisionAPI.Apply(prim)
    meshcollisionAPI = UsdPhysics.MeshCollisionAPI.Apply(prim)

    # meshcollisionAPI.CreateApproximationAttr().Set("sdf")
    # meshCollision = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)
    # meshCollision.CreateSdfResolutionAttr().Set(sdf_resolution)
    meshcollisionAPI.CreateApproximationAttr().Set("convexDecomposition")


# Start from the root prim and check all children recursively
for prim in stage.Traverse():
    if check_points_recursive(prim):
        print(f"Found 'points' in prim or its children: {prim.GetPath()}")
        target_prim = stage.GetPrimAtPath(prim.GetPath())
stage.Save()
