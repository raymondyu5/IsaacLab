from copy import deepcopy
from typing import List

import numpy as np
import sapien
import torch

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils import sapien_utils
from mani_skill.utils.structs.pose import vectorize_pose


@register_agent()
class LeapHandRight(BaseAgent):
    uid = "leap_hand_right"
    urdf_path = f"/home/ensu/Documents/weird/IsaacLab_assets/assets/robot/franka_leap/trash/franka_right_leap_long_finger.urdf"
    urdf_config = dict(
        _materials=dict(tip=dict(
            static_friction=2.0, dynamic_friction=1.0, restitution=0.0)),
        link={
            "link_3.0_tip":
            dict(material="tip", patch_radius=0.1, min_patch_radius=0.1),
            "link_7.0_tip":
            dict(material="tip", patch_radius=0.1, min_patch_radius=0.1),
            "link_11.0_tip":
            dict(material="tip", patch_radius=0.1, min_patch_radius=0.1),
            "link_15.0_tip":
            dict(material="tip", patch_radius=0.1, min_patch_radius=0.1),
        },
    )
    keyframes = dict(
        palm_side=Keyframe(
            qpos=np.zeros(16),
            pose=sapien.Pose([0, 0, 0.5], q=[1, 0, 0, 0]),
        ),
        palm_up=Keyframe(
            qpos=np.zeros(16),
            pose=sapien.Pose([0, 0, 0.5], q=[-0.707, 0, 0.707, 0]),
        ),
    )

    def __init__(self, *args, **kwargs):
        self.joint_names = [
            'panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4',
            'panda_joint5', 'panda_joint6', 'panda_joint7', 'j1', 'j12', 'j5',
            'j9', 'j0', 'j13', 'j4', 'j8', 'j2', 'j14', 'j6', 'j10', 'j3',
            'j15', 'j7', 'j11'
        ]

        self.joint_stiffness = 4e2
        self.joint_damping = 1e1
        self.joint_force_limit = 5e1

        # Order: thumb finger, index finger, middle finger, ring finger
        self.tip_link_names = [
            "link_15.0_tip",
            "link_3.0_tip",
            "link_7.0_tip",
            "link_11.0_tip",
        ]

        self.palm_link_name = "palm"
        super().__init__(*args, **kwargs)

    def _after_init(self):
        self.tip_links: List[sapien.Entity] = sapien_utils.get_objs_by_names(
            self.robot.get_links(), self.tip_link_names)
        self.palm_link: sapien.Entity = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.palm_link_name)

    @property
    def _controller_configs(self):
        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        joint_pos = PDJointPosControllerConfig(
            self.joint_names,
            None,
            None,
            self.joint_stiffness,
            self.joint_damping,
            self.joint_force_limit,
            normalize_action=False,
        )
        joint_delta_pos = PDJointPosControllerConfig(
            self.joint_names,
            -0.1,
            0.1,
            self.joint_stiffness,
            self.joint_damping,
            self.joint_force_limit,
            use_delta=True,
        )
        joint_target_delta_pos = deepcopy(joint_delta_pos)
        joint_target_delta_pos.use_target = True

        controller_configs = dict(
            pd_joint_delta_pos=joint_delta_pos,
            pd_joint_pos=joint_pos,
            pd_joint_target_delta_pos=joint_target_delta_pos,
        )

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)

    def get_proprioception(self):
        """
        Get the proprioceptive state of the agent.
        """
        obs = super().get_proprioception()
        # obs.update({
        #     "palm_pose":
        #     self.palm_pose,
        #     "tip_poses":
        #     self.tip_poses.reshape(-1,
        #                            len(self.tip_links) * 7),
        # })

        return obs

    @property
    def tip_poses(self):
        """
        Get the tip pose for each of the finger, four fingers in total
        """
        tip_poses = [
            vectorize_pose(link.pose, device=self.device)
            for link in self.tip_links
        ]
        return torch.stack(tip_poses, dim=-2)

    @property
    def palm_pose(self):
        """
        Get the palm pose for leap hand
        """
        return vectorize_pose(self.palm_link.pose, device=self.device)


@register_agent()
class LeapHandLeft(LeapHandRight):
    uid = "leap_hand_left"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/allegro/allegro_hand_left.urdf"


import sapien
from mani_skill.utils import sapien_utils, common
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env


@register_env("LeapPick-v1", max_episode_steps=200)
class LeapPickEnv(BaseEnv):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    SUPPORTED_ROBOTS = ["panda", "fetch", "leap_hand_right"]
    agent: [LeapHandRight, LeapHandLeft] = LeapHandRight

    def __init__(self, *args, robot_uids="leap_hand_right", **kwargs):
        # robot_uids="fetch" is possible, or even multi-robot
        # setups via robot_uids=("fetch", "panda")
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0, 0, 1]))

    def _load_scene(self, options: dict):
        # ...
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(
            # for boxes we specify half length of each side
            half_size=[0.02] * 3, )
        builder.add_box_visual(
            half_size=[0.02] * 3,
            material=sapien.render.RenderMaterial(
                # RGBA values, this is a red cube
                base_color=[1, 0, 0, 1], ),
        )
        # strongly recommended to set initial poses for objects, even if you plan to modify them later
        builder.initial_pose = sapien.Pose(p=[0, 0, 0.02], q=[1, 0, 0, 0])
        self.obj = builder.build(name="cube")
        # PushCube has some other code after this removed for brevity that
        # spawns a goal object (a red/white target) stored at self.goal_region

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # using torch.device context manager to auto create tensors
        # on CPU/CUDA depending on self.device, the device the env runs on
        with torch.device(self.device):
            # b = len(env_idx)
            # # use the TableSceneBuilder to init all objects in that scene builder
            # # self.table_scene.initialize(env_idx)

            # # here is randomization code that randomizes the x, y position
            # # of the cube we are pushing in the range [-0.1, -0.1] to [0.1, 0.1]
            # p = torch.zeros((b, 3))
            # p[..., :2] = torch.rand((b, 2)) * 0.2 - 0.1
            # p[..., 2] = self.cube_half_size
            # q = [1, 0, 0, 0]
            # obj_pose = Pose.create_from_pq(p=p, q=q)
            # self.obj.set_pose(obj_pose)
            pass

    def compute_normalized_dense_reward(self, obs, action, info):
        return 0.0


import gymnasium as gym

# Make the environment with GUI
env = gym.make("LeapPick-v1",
               robot_uids="leap_hand_right",
               render_mode="human")

# Reset environment (this sets up the scene)
obs = env.reset()
import matplotlib.pyplot as plt
# Loop to visualize for a few seconds
for _ in range(1000000):
    env.render_human()
    action = env.action_space.sample() * 0.0  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    # plt.imshow(env.render()[0].numpy())
