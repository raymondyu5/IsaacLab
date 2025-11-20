import numpy as np

from scripts.workflows.hand_manipulation.utils.dataset_utils.dexycb_dataset_viewer import DexYCBDataset
from scripts.workflows.hand_manipulation.utils.dataset_utils.hocap_dataset_viewer import HOCAPDataset
import torch
from scripts.workflows.utils.multi_datawrapper import MultiDatawrapper

from scripts.workflows.hand_manipulation.utils.dex_retargeting.retarget_utils import SynthesizeRobotPC, show_robot_mesh

import trimesh

import os


class DexRetargetingDataset:

    def __init__(self, args_cli, env_config, env, dataset_type="dexycb"):
        self.args_cli = args_cli
        self.env_config = env_config
        self.env = env
        self.device = env.unwrapped.device

        self.add_left_hand = args_cli.add_left_hand
        self.add_right_hand = args_cli.add_right_hand
        self.num_hand_joints = self.env_config["params"]["num_hand_joints"]

        if args_cli.data_type == "dexycb":
            dataset_class = DexYCBDataset
        elif args_cli.data_type == "hocap":
            dataset_class = HOCAPDataset

        if self.add_left_hand:
            self.hand_side = "left"
        elif self.add_right_hand:
            self.hand_side = "right"

        if self.args_cli.viz_hand:
            self.robot_sythesizer = SynthesizeRobotPC(env, env_config,
                                                      self.hand_side)

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

        if self.args_cli.save_data:
            self.collector_interface = MultiDatawrapper(
                args_cli,
                env_config,
                save_path=args_cli.save_path,
            )
            self.collector_interface.init_collector_interface()
        self.num_count = 0

    def analyze_data(self, manipulated_object_pose):
        lift_objects = []

        for object_name in manipulated_object_pose.keys():
            traj_data = torch.stack(manipulated_object_pose[object_name],
                                    dim=0)

            lift_or_not = (traj_data[-1, 2] - traj_data[0, 2]) > 0.10
            if lift_or_not:
                lift_objects.append(object_name)
        return lift_objects

    def save_data(self, target_qpos, manipulated_object_pose, robot_ee_pose,
                  lift_objects):
        if self.args_cli.save_data:

            obs_buffer = []

            for i in range(len(target_qpos)):
                obs_object_dict = {}

                for object_name in manipulated_object_pose.keys():
                    obs_object_dict[object_name] = manipulated_object_pose[
                        object_name][i].to(self.device).unsqueeze(0)
                    obs_object_dict["ee_pose"] = robot_ee_pose[i].to(
                        self.device).unsqueeze(0)

                obs_buffer.append(obs_object_dict)
                del obs_object_dict

            actions_buffer = target_qpos.unsqueeze(1)
            rewards_buffer = torch.zeros((len(target_qpos), 1),
                                         device=self.device)
            dones_buffer = torch.zeros((len(target_qpos), 1),
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

    def viz_robot(self, target_qpos, hand_side, ref_tip_value, object_name):

        last_qpos = torch.as_tensor(target_qpos[-1].to(self.device),
                                    device=self.device).unsqueeze(0)

        self.env.scene[f"{hand_side}_hand"].root_physx_view.set_dof_positions(
            last_qpos, indices=torch.arange(self.env.num_envs).to(self.device))

        for i in range(10):

            self.env.step(last_qpos)
        vertx, faces = self.robot_sythesizer.synthesize_pc()
        vertx = vertx[0].cpu().numpy()
        faces = faces[0].cpu().numpy()
        tip_joint = ref_tip_value[0]
        object_vertex = ref_tip_value[3]
        hand_object_faces = ref_tip_value[4]

        mano_hand_vertx = ref_tip_value[1]
        mano_hand_face = ref_tip_value[2]

        robot_hand_object_vertices = np.concatenate([vertx, object_vertex],
                                                    axis=0)
        robot_hand_object_faces = np.concatenate(
            [faces, hand_object_faces + len(vertx)], axis=0)

        mano_hand_object_vertices = np.concatenate(
            [mano_hand_vertx, object_vertex], axis=0)

        robot_hand_object_vertices[:, 1] += 0.4

        mano_hand_object_faces = np.concatenate(
            [mano_hand_face, hand_object_faces + len(mano_hand_vertx)], axis=0)

        hand_object_vertices = np.concatenate(
            [robot_hand_object_vertices, mano_hand_object_vertices], axis=0)
        hand_object_faces = np.concatenate([
            robot_hand_object_faces,
            mano_hand_object_faces + len(robot_hand_object_vertices)
        ],
                                           axis=0)

        # show_robot_mesh(hand_object_vertices, hand_object_faces, tip_joint)
        save_dir = os.path.join(self.args_cli.log_dir,
                                f"hand_object_mesh/{object_name}")
        os.makedirs(save_dir, exist_ok=True)
        num_pre_saved = len(os.listdir(save_dir))
        hand_object_mesh = trimesh.Trimesh(vertices=hand_object_vertices,
                                           faces=hand_object_faces)
        hand_object_mesh.export(save_dir + f"/{num_pre_saved}.glb")

    def run(self, ):
        if self.add_left_hand:
            hand_side = "left"
        elif self.add_right_hand:
            hand_side = "right"
        dexretarget_env = getattr(self, f"{hand_side}_dexretarget_env")
        target_qpos, manipulated_object_pose, robot_ee_pose, lift_object_name, ref_tip_value = dexretarget_env.retaget(
            self.num_count)
        if target_qpos is None:
            print("No data to save")
            import pdb
            pdb.set_trace()
            return None
        self.num_count += 1

        if target_qpos is None:
            return None
        lift_objects = self.analyze_data(manipulated_object_pose)
        self.save_data(target_qpos, manipulated_object_pose, robot_ee_pose,
                       lift_objects)

        if self.args_cli.viz_hand:
            self.viz_robot(target_qpos, hand_side, ref_tip_value,
                           lift_object_name)

        if not self.args_cli.save_data_only and not self.args_cli.viz_hand:

            self.step_manipulated_pose(target_qpos, hand_side,
                                       manipulated_object_pose)
