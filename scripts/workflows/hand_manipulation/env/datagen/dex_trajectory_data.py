import numpy as np

from scripts.workflows.hand_manipulation.utils.dataset_utils.dexycb_dataset_viewer import DexYCBDataset
# from scripts.workflows.hand_manipulation.utils.dataset_utils.hocap_dataset_viewer import HOCAPDataset
import torch
from scripts.workflows.utils.multi_datawrapper import MultiDatawrapper

from scripts.workflows.hand_manipulation.utils.dex_retargeting.retarget_utils import SynthesizeRobotPC, show_robot_mesh

import trimesh
import isaaclab.utils.math as math_utils
import os
from scripts.workflows.hand_manipulation.utils.dataset_utils.grab_dataset_viewer import GrabDataset

from scripts.workflows.hand_manipulation.utils.dataset_utils.meshviewer import MeshViewer, Mesh, colors
from scripts.workflows.hand_manipulation.utils.dataset_utils.visionpro_dataset_viewer import VisionProDataset

from manopth.manolayer import ManoLayer


class DexTrajDataset:

    def __init__(self, args_cli, env_config, env, dataset_type="dexycb"):
        self.args_cli = args_cli
        self.env_config = env_config
        self.env = env
        self.device = env.unwrapped.device

        self.add_left_hand = args_cli.add_left_hand
        self.add_right_hand = args_cli.add_right_hand
        self.num_hand_joints = self.env_config["params"]["num_hand_joints"]

        if args_cli.data_type == "grab":
            dataset_class = GrabDataset
        if args_cli.data_type == "dexycb":
            dataset_class = DexYCBDataset
        if args_cli.data_type == "visionpro":
            dataset_class = VisionProDataset

        if self.args_cli.viz_hand:
            if self.add_left_hand:
                self.left_robot_sythesizer = SynthesizeRobotPC(
                    env, env_config, "left")
            if self.add_right_hand:
                self.right_robot_sythesizer = SynthesizeRobotPC(
                    env, env_config, "right")
            # self.mesh_viewer = MeshViewer(width=1600,
            #                               height=1200,
            #                               offscreen=False)

        if self.add_left_hand:

            self.left_dexretarget_env = dataset_class(
                env_config=self.env_config,
                args_cli=args_cli,
                env=self.env,
                robot=self.env.scene["left_hand"],
                data_root=args_cli.data_dir,
            )
            self.num_data = self.left_dexretarget_env.num_data
        if self.add_right_hand:
            self.right_dexretarget_env = dataset_class(
                env_config=self.env_config,
                args_cli=args_cli,
                env=self.env,
                robot=self.env.scene["right_hand"],
                data_root=args_cli.data_dir,
            )
            self.num_data = self.right_dexretarget_env.num_data

        if self.args_cli.save_path is not None:
            self.collector_interface = MultiDatawrapper(
                args_cli,
                env_config,
                save_path=args_cli.save_path,
            )

        self.num_count = 0

    def save_data(self, retarget_info):

        obs_buffer = []

        # for i in range(retarget_info["num_frame"]):
        num_frame = retarget_info["num_frame"]
        if "rhand_verts" in retarget_info.keys(
        ) or "lhand_verts" in retarget_info.keys():
            if self.add_right_hand:
                rhand_verts = retarget_info["rhand_verts"]

                rhand_joints = retarget_info["rhand_joints"]
            if self.add_left_hand:
                # lhand_faces = retarget_info["lhand_faces"]
                # object_verts = retarget_info["object_verts"]
                lhand_verts = retarget_info["lhand_verts"]
                # object_faces = retarget_info["object_faces"]
                lhand_joints = retarget_info["lhand_joints"]
                # object_transformation = retarget_info["object_transformation"]

            for index in range(num_frame):
                obs_object_dict = {}

                # object_name = retarget_info["object_name"]
                # obs_object_dict["object_verts"] = object_verts[index]
                if self.add_right_hand:
                    obs_object_dict["rhand_verts"] = rhand_verts[index]
                    obs_object_dict["rhand_joints"] = rhand_joints[index]

                if self.add_left_hand:
                    obs_object_dict["lhand_verts"] = lhand_verts[index]
                    obs_object_dict["lhand_joints"] = lhand_joints[index]

                obs_buffer.append(obs_object_dict)

        actions_buffer = []
        if self.add_left_hand:

            actions_buffer.append(retarget_info["retarget_left_joints"])
        if self.add_right_hand:

            actions_buffer.append(retarget_info["retarget_right_joints"])

        actions_buffer = torch.cat(actions_buffer, dim=1)

        rewards_buffer = torch.zeros((retarget_info["num_frame"], 1),
                                     device=self.device)
        dones_buffer = torch.zeros((retarget_info["num_frame"], 1),
                                   device=self.device)
        self.collector_interface.add_demonstraions_to_buffer(
            obs_buffer,
            actions_buffer,
            rewards_buffer,
            dones_buffer,
        )

    def step_manipulated_pose(self, target_qpos, hand_side,
                              manipulated_object_pose):

        init_qpos = torch.as_tensor(target_qpos[0].to(self.device),
                                    device=self.device).unsqueeze(0)

        self.env.scene[f"{hand_side}_hand"].root_physx_view.set_dof_positions(
            init_qpos, indices=torch.arange(self.env.num_envs).to(self.device))

        for i in range(10):

            for object_name in self.env.scene.rigid_objects.keys():

                if object_name not in manipulated_object_pose.keys():
                    continue
                object_pose = manipulated_object_pose[object_name][0].to(
                    self.device)

                object_pose = object_pose.unsqueeze(0)

                self.env.scene.rigid_objects[
                    object_name].write_root_pose_to_sim(
                        object_pose.unsqueeze(0),
                        torch.arange(self.env.num_envs).to(self.device))
            self.env.step(init_qpos)

        for i in range(len(target_qpos)):

            actions = torch.zeros(self.env.action_space.shape,
                                  device=self.device)

            actions = torch.as_tensor(target_qpos[i].to(self.device),
                                      device=self.device).unsqueeze(0)

            obs, rewards, terminated, time_outs, extras = self.env.step(
                actions)

        self.env.reset()

    def viz_robot(self, info, index, hand_side="right"):

        last_qpos = torch.as_tensor(
            info[f"retarget_{hand_side}_joints"][index],
            device=self.device).unsqueeze(0).to(self.device)

        self.env.scene[f"{hand_side}_hand"].root_physx_view.set_dof_positions(
            last_qpos, indices=torch.arange(self.env.num_envs).to(self.device))

        for i in range(100):

            self.env.step(last_qpos)
        self.render_robot(info, index, hand_side=hand_side)

    def fit_mano_hand(self, target_joints):

        batch_size = 1
        # Select number of principal components for pose space
        ncomps = 6

        mano_layer = ManoLayer(
            mano_root=
            'scripts/workflows/hand_manipulation/utils/manopth/mano_v1_2/models',
            use_pca=True,
            flat_hand_mean=True,
            side="right" if self.add_right_hand else "left",
            ncomps=ncomps)

        shape_param = torch.rand(batch_size,
                                 10,
                                 requires_grad=True,
                                 device='cpu')
        # Generate random pose parameters, including 3 values for global axis-angle rotation
        pose_param = torch.rand(batch_size,
                                ncomps + 3,
                                requires_grad=True,
                                device='cpu')

        # Step 4: Setup optimizer
        optimizer = torch.optim.Adam([pose_param, shape_param], lr=0.01)

        # Step 5: Fit loop
        for i in range(5000):  # increase if needed
            optimizer.zero_grad()
            verts, joints = mano_layer(pose_param,
                                       shape_param)  # (1, 778, 3), (1, 21, 3)

            loss = torch.nn.functional.mse_loss(joints, target_joints * 1000)
            loss.backward()
            optimizer.step()

            if i % 50 == 0:
                print(f"Step {i}: Loss = {loss.item():.4f}")
        from scripts.workflows.hand_manipulation.utils.dex_retargeting.retarget_utils import get_hand_joint_names, init_leap_hand_retarget, display_hand
        import pdb
        pdb.set_trace()
        # Step 6: Visualize the fitted hand mesh
        display_hand({
            "hand_info": {
                'verts': verts[0].detach().cpu(),
                'joints': joints[0].detach().cpu(),
                "faces": mano_layer.th_faces
            },
        })

    def render_robot(self, info, index, hand_side="right"):

        if hand_side == "left":
            robot_vertx, robot_faces = self.left_robot_sythesizer.synthesize_pc(
            )
        elif hand_side == "right":
            robot_vertx, robot_faces = self.right_robot_sythesizer.synthesize_pc(
            )

        if "rhand_verts" not in info.keys() and "lhand_verts" not in info.keys(
        ):

            # self.fit_mano_hand(info["rhand_joints"][index].unsqueeze(0))
            hand_vertices = robot_vertx
            hand_faces = robot_faces

            show_robot_mesh(
                hand_vertices[0].cpu().numpy(),
                hand_faces[0].cpu().numpy(),
            )

        else:

            robot_vertx = robot_vertx[0].cpu().numpy()
            robot_faces = robot_faces[0].cpu().numpy()
            rhand_vertices = info["rhand_verts"][index]  #.cpu().numpy()
            rhand_faces = info["rhand_faces"]
            robot_vertx[..., 0] += 0.4
            hand_vertices = np.concatenate([robot_vertx, rhand_vertices],
                                           axis=0)
            hand_faces = np.concatenate(
                [robot_faces, rhand_faces + robot_vertx.shape[0]], axis=0)
            show_robot_mesh(hand_vertices, hand_faces,
                            info["rhand_joints"][index].cpu().numpy())

        # rhand_mesh = Mesh(vertices=rhand_vertices,
        #                   faces=rhand_faces,
        #                   vc=colors['pink'],
        #                   smooth=True)

        # robot_mesh = Mesh(vertices=robot_vertx,
        #                   faces=robot_faces,
        #                   vc=colors['grey'],
        #                   smooth=True)
        # self.mesh_viewer.set_static_meshes([rhand_mesh, robot_mesh])

        # object_name = info["object_name"]

        # save_dir = os.path.join(self.args_cli.log_dir,
        #                         f"hand_object_mesh/{object_name}")

        # num_pre_saved = len(os.listdir((save_dir))) - 1

        # hand_object_mesh = trimesh.Trimesh(vertices=hand_vertices,
        #                                    faces=hand_faces)
        # hand_object_mesh.export(save_dir + f"/{num_pre_saved}/{index}.glb")

    def reset(self, retarget_info, hand_side="right"):
        self.env.reset()

        # object_name = retarget_info["object_name"]

        # save_dir = os.path.join(self.args_cli.log_dir,
        #                         f"hand_object_mesh/{object_name}")
        # os.makedirs(save_dir, exist_ok=True)

        # num_pre_saved = len(os.listdir((save_dir)))
        # os.makedirs(save_dir + f"/{num_pre_saved}", exist_ok=True)

        last_qpos = torch.as_tensor(
            retarget_info[f"retarget_{hand_side}_joints"][0].to(self.device),
            device=self.device).unsqueeze(0)

        self.env.scene[f"{hand_side}_hand"].root_physx_view.set_dof_velocities(
            last_qpos, indices=torch.arange(self.env.num_envs).to(self.device))

        for i in range(20):

            self.env.step(last_qpos)

    def run(self, ):
        if self.add_left_hand:
            hand_side = "left"
        elif self.add_right_hand:
            hand_side = "right"
        dexretarget_env = getattr(self, f"{hand_side}_dexretarget_env")

        retarget_info = dexretarget_env.retaget(self.num_count)
        if retarget_info is None:
            return

        # for i in range(0, 100, 20):
        #     try:
        #         self.viz_robot(retarget_info, index=i, hand_side=hand_side)
        #     except:
        #         pass

        num_frame = retarget_info["num_frame"]
        self.num_count += 1

        if self.args_cli.save_path is not None:
            self.save_data(retarget_info)

        if not self.args_cli.save_data_only:
            self.reset(retarget_info, hand_side=hand_side)
            for index in range(0, num_frame):
                actions = []
                if self.add_left_hand:
                    actions.append(
                        retarget_info["retarget_left_joints"][index].to(
                            self.device))
                if self.add_right_hand:
                    actions.append(
                        retarget_info["retarget_right_joints"][index].to(
                            self.device))

                actions = torch.cat(actions, dim=0).unsqueeze(0)
                # actions[:, [10, 12, 13]] = 0.0
                self.env.step(actions)

                # self.render_robot(
                #     retarget_info,
                #     index,
                # )
