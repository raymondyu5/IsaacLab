import torch
import isaaclab.utils.math as math_utils
import sys
from typing import Union
from pxr import Sdf, Usd
import weakref
import warnings
import logging
import gym
from tools.curobo_planner import IKPlanner
import yaml
import numpy as np
from pxr import PhysxSchema, UsdPhysics
from tools.curobo_planner import IKPlanner
from isaaclab.sim import converters, schemas
import isaaclab.sim as sim_utils
# from omni.isaac.robot_assembler import RobotAssembler, AssembledRobot
from tools.trash.robot_assembler import RobotAssembler


def assemble_soft_rigid(env, object, env_ids):

    if object.cfg.deform_cfg["attach_robot"][
            "init"] and not object.cfg.deform_cfg["attach_robot"]["attact_yet"]:

        for i in env_ids.cpu().numpy():

            attachment_link = PhysxSchema.PhysxPhysicsAttachment.Define(
                env.scene.stage,
                object.mesh_path_expr.replace(".*", str(i)) + "/attachment")
            attachment_link.GetActor0Rel().SetTargets(
                [object.mesh_path_expr.replace(".*", str(i))])

            attach_object_path = object.cfg.deform_cfg["attach_robot"][
                "attach_object_path"]

            attachment_link.GetActor1Rel().SetTargets(
                [attach_object_path.replace(".*", str(i))])
            # attachment_link.GetActor1Rel().SetTargets(
            #     ["/World/envs/env_" + str(0) + "/Robot/panda_hand"])

            autoApi = PhysxSchema.PhysxAutoAttachmentAPI.Apply(
                attachment_link.GetPrim())
            autoApi.GetCollisionFilteringOffsetAttr().Set(0.04)
            autoApi.GetDeformableVertexOverlapOffsetAttr().Set(
                object.cfg.deform_cfg["attach_robot"]
                ["DeformableVertexOverlapOffset"])
            autoApi.GetEnableDeformableVertexAttachmentsAttr().Set(1)
            autoApi.GetEnableRigidSurfaceAttachmentsAttr().Set(1)

            schemas.modify_mass_properties(
                object.mesh_path_expr.replace(".*", str(i)),
                sim_utils.MassPropertiesCfg(mass=float(100)))
            # schemas.modify_mass_properties(
            #     attach_object_path.replace(".*", str(i)),
            #     sim_utils.MassPropertiesCfg(mass=float(0.0)))
        object.cfg.deform_cfg["attach_robot"]["attact_yet"] = True
        return True
    else:
        return False


class SweepEnv:

    def __init__(self, env, init_pos):
        self.env = weakref.proxy(env)
        self.num_envs = env.num_envs
        self.device = env.device
        self.env_indices = torch.arange(self.num_envs,
                                        dtype=torch.int64,
                                        device=self.device)
        self.stage = self.env.scene.stage
        if "robot" in self.env.scene.keys():
            self.init_robot(init_pos)
        # self.rigid_attachment("sweep01")
        assemble_soft_rigid(self.env, self.env.scene["broom01"],
                            self.env_indices)
        self.init_pos = init_pos

        # step one for update
        self.step_env()

    def init_robot(self, init_pos):

        curobo_ik = IKPlanner()
        target_pos = init_pos.clone()
        target_pos[:, 2] += 0.3
        init_qpos = curobo_ik.plan_motion(target_pos[:, :3], target_pos[:, 3:])
        self.robot_target_qpos = self.env.scene[
            "robot"]._data.default_joint_pos[:, :9].clone()
        self.env.scene[
            "robot"]._data.default_joint_pos[:, :9] = init_qpos.squeeze(1)

        self.robot_target_qpos[:] = init_qpos.squeeze(1)

    def step_env(self):

        for k in range(self.env.max_episode_length):
            #sample actions from -1 to 1
            actions = torch.rand(self.env.action_space.shape,
                                 device=self.device) * 0
            actions[:, :7] = self.init_pos
            if k > 20:
                # actions[:, 0] += (k - 20) * 0.02
                actions[:, -2:] = -1
            if k > 50:
                actions[:, 0] += 0.005 * (k - 50)

            next_obs, reward, terminate, time_out, info = self.env.step(
                actions)

        self.env.scene["robot"].root_physx_view.set_dof_positions(
            self.robot_target_qpos, self.env_indices)
        self.env.scene["robot"].root_physx_view.set_dof_velocities(
            self.robot_target_qpos * 0, self.env_indices)

    def rigid_attachment(self, name):

        object = self.env.scene[name]
        target_attachment_path = object.cfg.deform_cfg["attach_robot"][
            "attach_object_path"]

        for i in self.env_indices.cpu().numpy():

            robot_assembler = RobotAssembler()
            assembled_robot = robot_assembler.assemble_articulations(
                f"/World/envs/env_{i}/Robot",
                target_attachment_path.replace(".*", str(i)),
                "/panda_hand",
                "",
                np.array([0.0, 0.0, 0.0]),
                np.array([1.0, 0.0, 0.0, 0.0]),
                mask_all_collisions=True,
                single_robot=False)
