import torch as th
import torch.nn as nn

import torch


def load_dinov3(name='dinov3_vits16'):
    """
    Load dinov3 model
    """

    REPO_DIR = "submodule/dinov3"  # your local repo (with hubconf.py)
    if name == 'dinov3_vits16':

        CKPT_PATH = "submodule/dinov3/ckpt/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

    model = torch.hub.load(repo_or_dir=REPO_DIR,
                           model="dinov3_vits16",
                           source="local",
                           weights=CKPT_PATH).cuda()
    return model


class NatureCNN:

    def __init__(
        self,
        input_image_size: tuple[int, int, int] = (3, 84, 84),
        features_dim: int = 256,
    ) -> None:

        self.cnn = nn.Sequential(
            nn.Conv2d(input_image_size[0],
                      32,
                      kernel_size=8,
                      stride=4,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            sample_input = th.zeros(1, *input_image_size)
            n_flatten = self.cnn(sample_input).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim),
                                    nn.ReLU())

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(x))
