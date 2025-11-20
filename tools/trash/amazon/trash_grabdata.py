import torch
import smplx
import numpy as np
from manopth import demo
from manopth.manolayer import ManoLayer


def prepare_params(params, dtype=np.float32):
    return {k: v.astype(dtype) for k, v in params.items()}


def params2torch(params, dtype=torch.float32):
    return {k: torch.from_numpy(v).type(dtype) for k, v in params.items()}


data = np.load(
    "/media/ensu/data/datasets/grab/hand_dataset/s1/airplane_fly_1.npz",
    allow_pickle=True)

# --- 1. Load the MANO Model (right hand) ---
# Download mano models if needed: https://mano.is.tue.mpg.de/download.php
import trimesh
from scripts.workflows.hand_manipulation.utils.dex_retargeting.mano_layer import MANOLayer

sbj_mesh = "/media/ensu/data/datasets/grab/people/male/s1_rhand.ply"
rh_vtemp = np.array(trimesh.load(sbj_mesh).vertices, dtype=np.float32)

hand_shape = torch.as_tensor(
    np.load("/media/ensu/data/datasets/grab/people/male/s1_rhand_betas.npy"))
hand_finger_info = data["rhand"].item()["params"]
rhand_finger_pose = np.concatenate(
    [hand_finger_info["global_orient"], hand_finger_info["hand_pose"]], axis=1)

mano_layer = ManoLayer(mano_root='/media/ensu/data/datasets/smplx/models/mano',
                       use_pca=True,
                       flat_hand_mean=True,
                       ncomps=24)

hand_mesh = trimesh.Trimesh(
    vertices=mano_layer.th_v_template[0].detach().cpu().numpy(),
    faces=mano_layer.th_faces.detach().cpu().numpy())
# hand_mesh.export("test_hand.ply")

mano_layer.th_v_template[0] = torch.as_tensor(rh_vtemp)
hand_verts, hand_joints = mano_layer(
    torch.as_tensor(rhand_finger_pose)[800].unsqueeze(0).to(
        dtype=torch.float32),
    hand_shape.unsqueeze(0).to(dtype=torch.float32),
)

demo.display_hand(
    {
        'verts': hand_verts.detach(),
        'joints': hand_joints.detach(),
    },
    mano_faces=mano_layer.th_faces)
hand_info = params2torch(prepare_params(data["rhand"].item()["params"]))

smplx_model = smplx.create(
    model_path=
    '/media/ensu/data/datasets/smplx/models',  # folder containing MANO_RIGHT.pkl
    model_type='mano',
    gender="male",
    num_pca_comps=24,
    is_rhand=True,
    v_template=rh_vtemp,
    flat_hand_mean=True,
    batch_size=1113  # not using PCA
)

result = smplx_model(**hand_info)
demo.display_hand({
    'verts': result.vertices[300].unsqueeze(0).detach(),
    'joints': result.joints[300].unsqueeze(0).detach()
})
