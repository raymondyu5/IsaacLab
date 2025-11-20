import numpy as np
import trimesh
from PIL import Image
import os
import pickle

from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": True})
from pxr import Gf, Kind, Sdf, Usd, UsdGeom, UsdShade, Vt, UsdPhysics, UsdLux
import transformations

from scipy.spatial.transform import Rotation
with open(
        "/media/lme/data4/weird/IsaacLab/tools/auto_articulation/asset/drawers/drawer_0.pkl",
        'rb') as f:
    drawer_info = pickle.load(f)

drawer_transform = drawer_info["transform"]
interact_type = drawer_info["interact"]
