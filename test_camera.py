"""Camera demo - working version."""

from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"headless": True, "enable_cameras": True})
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.sensors import Camera, CameraCfg
from pxr import Gf, UsdGeom, UsdLux
from PIL import Image
import numpy as np

# Create simulation
sim = sim_utils.SimulationContext()
stage = sim.stage

# Create ground manually
ground = UsdGeom.Cube.Define(stage, "/World/Ground")
ground.GetSizeAttr().Set(10.0)
ground.AddTranslateOp().Set(Gf.Vec3f(0, 0, -5))

# Create red cube
cube = UsdGeom.Cube.Define(stage, "/World/RedCube")
cube.GetSizeAttr().Set(0.5)
cube.AddTranslateOp().Set(Gf.Vec3f(0, 0, 0.5))
cube.GetDisplayColorAttr().Set([Gf.Vec3f(1, 0, 0)])

# Add dome light
light = UsdLux.DomeLight.Define(stage, "/World/Light")
light.CreateIntensityAttr(3000.0)

# Create camera (spawn it with position/orientation)
camera_cfg = CameraCfg(
    prim_path="/World/Camera",
    update_period=0.1,
    height=480,
    width=640,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0,
        horizontal_aperture=20.955,
    ),
)
camera = Camera(cfg=camera_cfg)

# Manually position the camera prim before reset (get existing ops, don't add new ones)
camera_prim = stage.GetPrimAtPath("/World/Camera")
xformable = UsdGeom.Xformable(camera_prim)
translate_op = xformable.GetOrderedXformOps()[0]  # existing translate op
orient_op    = xformable.GetOrderedXformOps()[1]  # existing orient op

# --- NEW POSE: place camera behind + above cube and look at it ---

# Camera position (x, y, z)
# A bit behind the cube along -Y and slightly above
translate_op.Set(Gf.Vec3d(0.0, -3.0, 1.5))

# Orientation quaternion (w, x, y, z)
# This rotates the camera so its forward axis (-Z) points toward the cube at (0, 0, 0.5)
orient_op.Set(Gf.Quatd(0.81124219, 0.58471028, 0.0, 0.0))


# Reset simulation - this initializes the camera
sim.reset()
print("✓ Camera initialized")

# Run simulation
for i in range(100000):
    sim.step()
    camera.update(dt=sim.get_physics_dt())
    if i % 100 == 0:
        rgb = camera.data.output["rgb"]    # torch tensor, (H,W,4) uint8

        # Convert to numpy and drop alpha channel
        # breakpoint()
        rgb_np = rgb[0, ..., :3].cpu().numpy()
        # Save using PIL
        img = Image.fromarray(rgb_np, mode="RGB")
        img.save(f"frame_{i:05d}.png")

        print(f"Saved frame_{i:05d}.png")


print("\n✓ Camera test successful!")
simulation_app.close()
