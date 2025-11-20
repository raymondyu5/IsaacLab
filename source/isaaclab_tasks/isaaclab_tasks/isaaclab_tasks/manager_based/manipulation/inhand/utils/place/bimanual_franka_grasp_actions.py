from isaaclab.envs.mdp.actions.joint_actions import JointAction
from isaaclab.envs.mdp.actions import actions_cfg
from isaaclab.envs import ManagerBasedEnv


class HandActions(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    cfg: actions_cfg.JointPositionActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.JointPositionActionCfg,
                 env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        self.env = env
        # use default joint positions as offset
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[:, self.
                                                              _joint_ids].clone(
                                                              )

    def apply_actions(self):
        # set position targets

        self._asset.set_joint_position_target(self.processed_actions,
                                              joint_ids=self._joint_ids)
