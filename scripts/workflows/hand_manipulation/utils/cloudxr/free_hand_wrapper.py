import torch
import isaaclab.utils.math as math_utils


class FreeHandWrapper:

    def __init__(self, env, teleop_interface, add_left_hand, add_right_hand):
        self.teleop_interface = teleop_interface
        self.env = env
        self.add_left_hand = add_left_hand
        self.add_right_hand = add_right_hand

        self.left_hand_quat_offset = torch.tensor([
            .0000, 0.7070, -0.7070, 0.0000
        ]).to(device=self.env.device).unsqueeze(0).repeat_interleave(
            self.env.num_envs, dim=0)
        self.right_hand_quat_offset = torch.tensor([
            .0000, 0.7070, 0.7070, 0.0000
        ]).to(device=self.env.device).unsqueeze(0).repeat_interleave(
            self.env.num_envs, dim=0)

        self.hand_pos_offset = torch.tensor(
            [0.20, 0.0,
             0.0]).to(device=self.env.device).unsqueeze(0).repeat_interleave(
                 self.env.num_envs, dim=0)

    def step_teleoperation(self, teleoperation_active,
                           reset_stopped_teleoperation):

        teleop_data, _ = self.teleop_interface.advance()

        if teleoperation_active:

            self.env.step(
                torch.as_tensor(teleop_data[0][-1]).to(
                    self.env.device).unsqueeze(0))

            if self.add_left_hand:
                default_left_hand = self.env.scene[
                    "left_hand"]._data.reset_root_state.clone()

                default_left_hand[..., 3:7] = math_utils.quat_mul(
                    torch.as_tensor(
                        teleop_data[0][0][3:7]).unsqueeze(0).repeat_interleave(
                            self.env.num_envs, dim=0),
                    self.left_hand_quat_offset)
                default_left_hand[..., :3] = torch.as_tensor(
                    teleop_data[0][0][:3]).unsqueeze(0).repeat_interleave(
                        self.env.num_envs, dim=0) + self.hand_pos_offset

                self.env.scene["left_hand"].write_root_pose_to_sim(
                    default_left_hand[..., :7],
                    torch.arange(self.env.num_envs).to(self.env.device))
            if self.add_right_hand:
                default_right_hand = self.env.scene[
                    "right_hand"]._data.reset_root_state.clone()

                default_right_hand[..., 3:7] = math_utils.quat_mul(
                    torch.as_tensor(
                        teleop_data[0][1][3:7]).unsqueeze(0).repeat_interleave(
                            self.env.num_envs, dim=0),
                    self.right_hand_quat_offset)
                default_right_hand[..., :3] = torch.as_tensor(
                    teleop_data[0][1][:3]).unsqueeze(0).repeat_interleave(
                        self.env.num_envs, dim=0) + self.hand_pos_offset

                self.env.scene["right_hand"].write_root_pose_to_sim(
                    default_right_hand[..., :7],
                    torch.arange(self.env.num_envs).to(self.env.device))

        else:
            self.env.sim.render()

        if reset_stopped_teleoperation:
            self.env.reset()
            reset_stopped_teleoperation = False
            print("Resetting environment...")
