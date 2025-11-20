import numpy as np
import torch
from scripts.workflows.hand_manipulation.utils.meshcatviewer.button_wrapper import ButtonWrapper


class VAESliderApp:

    def __init__(self,
                 raw_sliders,
                 root,
                 action_buffer=None,
                 vae_path=None,
                 hand_side="right",
                 device="cpu",
                 slider_function=None,
                 group_count=0,
                 num_slider=1):

        # Define joint names and dimensions
        self.raw_joint_names = [
            'j1', 'j0', 'j2', 'j3', 'j12', 'j13', 'j14', 'j15', 'j5', 'j4',
            'j6', 'j7', 'j9', 'j8', 'j10', 'j11'
        ]
        self.isaac_joint_names = [
            'j1', 'j12', 'j5', 'j9', 'j0', 'j13', 'j4', 'j8', 'j2', 'j14',
            'j6', 'j10', 'j3', 'j15', 'j7', 'j11'
        ]
        self.retarget2pin = [
            self.isaac_joint_names.index(name) for name in self.raw_joint_names
        ]
        self.num_hand_joints = len(self.raw_joint_names)
        self.vae_path = vae_path
        self.hand_side = hand_side
        self.device = device
        self.raw_sliders = raw_sliders
        self.root = root
        self.slider_function = slider_function
        self.action_buffer = action_buffer
        self.num_demos = len(self.action_buffer) if action_buffer else 0

        self.group_count = self.load_vae(group_count)
        if self.action_buffer is not None:
            self.group_count = self.init_reconstruct_slides(
                self.group_count, num_slider)
        else:
            self.group_count = group_count

    def init_reconstruct_slides(self, group_count, num_slider):
        self.button_wrapper = ButtonWrapper(self.root,
                                            num_demos=self.num_demos,
                                            raw_sliders=self.raw_sliders)
        group_count = self.button_wrapper.create_demo_control_panel(
            group_count, num_slider, name="VAE Reconstruction")
        self.recontruct_vae_actions = [[] for _ in range(len(self.vae_models))]
        for index, vae_model in enumerate(self.vae_models):
            with torch.no_grad():
                for action in self.action_buffer:
                    if vae_model.data_normalizer is not None:
                        normalized_action = vae_model.normalize_action(
                            torch.as_tensor(
                                action[...,
                                       -self.num_hand_joints:]).to("cuda").to(
                                           dtype=torch.float32))
                        enconded_action, _ = vae_model.model.encoder(
                            None, normalized_action)
                    else:
                        enconded_action, _ = vae_model.model.encoder(
                            None,
                            torch.as_tensor(
                                action[...,
                                       -self.num_hand_joints:]).to("cuda").to(
                                           dtype=torch.float32))
                    decoded_action = vae_model.model.decoder(
                        None, enconded_action)[0].cpu().numpy()
                    # self.recontruct_vae_actions.append([
                    #     decoded_action[:, self.retarget2pin].tolist(),
                    #     action[:, -self.num_hand_joints:]
                    #     [:, self.retarget2pin].tolist()
                    # ])
                    self.recontruct_vae_actions[index].append(decoded_action)

        return group_count

    def load_vae(self, group_count):
        from scripts.workflows.hand_manipulation.utils.vae.vae_family import VAEFAMILY
        self.vae_sliders = []
        self.vae_models = []
        self.vae_model_range = []

        for index, vae_path in enumerate(self.vae_path):
            vae_sliders = []

            vae_model = VAEFAMILY(None,
                                  eval=True,
                                  hand_side=self.hand_side,
                                  vae_path=vae_path)

            model_type = vae_model.model_type
            self.vae_models.append(vae_model)
            self.vae_model_range.append([
                np.array(vae_model.min_latent_value),
                np.array(vae_model.max_latent_value)
            ])

            self.slider_function(
                f"{model_type}_dim_{vae_model.latent_dim}",
                [f"z[{i}]"
                 for i in range(vae_model.latent_dim)], -1, 1, vae_sliders)
            self.vae_sliders.append(vae_sliders)
            group_count += 1
        return group_count

    def get_vae_values(self):

        actions = []
        vae_reconstructed_actions = {"vae_reconstructed": []}
        for index, vae_model in enumerate(self.vae_models):
            vae_slider = self.vae_sliders[index]

            vae_values = np.array([v.get() for v in vae_slider])
            action_range = self.vae_model_range[index]

            vae_values = (vae_values + 1) / 2 * (
                action_range[1] - action_range[0]) + action_range[0]

            vae_values = torch.as_tensor(vae_values).unsqueeze(0).to(
                "cuda").to(dtype=torch.float32)
            with torch.no_grad():
                reconstructed_actions = vae_model.decoder(
                    None, vae_values).cpu().numpy()
            actions.append(
                reconstructed_actions[0][self.retarget2pin].tolist())

            if self.action_buffer is not None:
                target_demo_data = self.action_buffer[
                    self.button_wrapper.demo_count % self.num_demos]
                num_frames = len(target_demo_data)

                recontruct_vae_action = self.recontruct_vae_actions[index][
                    self.button_wrapper.demo_count %
                    self.num_demos][self.button_wrapper.frame_index %
                                    num_frames][self.retarget2pin]
                raw_action = self.action_buffer[
                    self.button_wrapper.demo_count % self.num_demos][
                        self.button_wrapper.frame_index %
                        num_frames][...,
                                    -self.num_hand_joints:][self.retarget2pin]

                vae_reconstructed_actions["vae_reconstructed"].append(
                    np.concatenate([recontruct_vae_action,
                                    raw_action]).tolist())

                if self.button_wrapper.play_demo:

                    self.button_wrapper.frame_index += 1

        return {"vae": actions} | vae_reconstructed_actions
