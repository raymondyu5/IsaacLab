from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
import torchvision.transforms as T
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from scripts.workflows.hand_manipulation.utils.diffusion.dataset.utils import ColorRandomizer, GaussianNoiseRandomizer
import torch.nn as nn


class DinoV2Encoder(nn.Module):

    def __init__(self, base_model, feature_key, device="cuda"):
        super().__init__()

        self.base_model = base_model

        self.device = device
        self.feature_key = feature_key
        self.emb_dim = self.base_model.num_features
        if feature_key == "x_norm_patchtokens":
            self.latent_ndim = 2
        elif feature_key == "x_norm_clstoken":
            self.latent_ndim = 1
        else:
            raise ValueError(f"Invalid feature key: {feature_key}")

        self.patch_size = self.base_model.patch_size

    def forward(self, x):
        emb = self.base_model.forward_features(x)[self.feature_key]
        if self.latent_ndim == 1:
            emb = emb.unsqueeze(1)  # dummy patch dim
        return emb


from tools.trash.amazon.trash_image_bugs import load_diffusion_model

diffusion_path = "logs/trash/image_cfm/"
device = "cuda:0"
model, cfg = load_diffusion_model(diffusion_path, device)

dino = model.obs_encoder.key_model_map.rgb_0

dino_encoder = DinoV2Encoder(dino, "x_norm_patchtokens")

transform = T.Compose([
    T.Resize((1120, 1120), interpolation=T.InterpolationMode.BICUBIC),
    # T.CenterCrop(560),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
image = np.array(Image.open('/home/ensu/Downloads/rgb.png').convert('RGB'))


def visualize_dino_features(
    raw_images,
    threshold=0.2,
    batch_pca=True,
):
    """
    raw_images: numpy array of shape (B, H, W, 3) or (B, 3, H, W)
    Values are assumed to be in [0, 1] float range.
    """
    batch_size = raw_images.shape[0]

    transformed_images = []
    for i in range(batch_size):
        img_np = raw_images[i]

        # Convert (3, H, W) to (H, W, 3) if needed
        if img_np.shape[0] == 3 and img_np.shape[-1] != 3:
            img_np = np.transpose(img_np, (1, 2, 0))

        # Convert to uint8 PIL Image
        img_pil = Image.fromarray((img_np).astype(np.uint8))

        # Apply DINOv2 transform
        img_tensor = transform(img_pil)
        transformed_images.append(img_tensor)

    # Stack into a batch
    transformed_batch = torch.stack(transformed_images).to(
        dino_encoder.device)  # (B, 3, H, W)

    with torch.no_grad():
        features = dino_encoder(transformed_batch)  # (B, N_patch, D)

    B, N_patch, D = features.shape
    pca = PCA(n_components=3)
    pca_features = []
    norm_features = []
    norm_features_foreground = []
    pca3 = PCA(n_components=3)
    scaler = MinMaxScaler()

    if not batch_pca:

        for i in range(B):
            feature = features[i].cpu().numpy()

            pca.fit(feature)
            pca_feature = pca.transform(feature)
            norm_feature = scaler.fit_transform(pca_feature)
            dim = int(np.sqrt(norm_feature.shape[0]))
            pca_features.append(pca_feature.reshape(dim, dim, -1))
            background = norm_feature > threshold

            norm_features.append(norm_feature.reshape(dim, dim, -1))

            bg_feature = feature.copy()

            bg_feature[background[:, 0]] = 0

            pca3.fit(bg_feature)
            features_foreground = pca3.transform(bg_feature)
            norm_feature_foreground = scaler.fit_transform(features_foreground)

            norm_features_foreground.append(
                np.where(background, 0, norm_feature).reshape(dim, dim, -1))

        pca_features = np.concatenate(pca_features, axis=1)
        norm_features = np.concatenate(norm_features, axis=1)
        norm_features_foreground = np.concatenate(norm_features_foreground,
                                                  axis=1)
    else:

        _, n_patch, dim = features.shape
        features = features.reshape(batch_size * n_patch, dim)
        features = features.cpu()
        pca.fit(features)
        pca_features = pca.transform(features)
        norm_features = scaler.fit_transform(pca_features)

        background = norm_features > threshold  #Adjust threshold based on your images

        bg_features = features.clone()  #make a copy of features
        for i in range(bg_features.shape[-1]):
            bg_features[:, i][background[:, 0]] = 0

        pca3.fit(bg_features)
        features_foreground = pca3.transform(bg_features)
        norm_features_foreground = scaler.fit_transform(features_foreground)
        dim = int(np.sqrt(n_patch))

        norm_features_foreground = norm_features_foreground.reshape(
            batch_size, dim, dim, 3)
        pca_features = pca_features.reshape(batch_size, dim, dim)
        norm_features = norm_features.reshape(batch_size, dim, dim)
        norm_features_foreground = norm_features_foreground.reshape(
            batch_size, dim, dim, 3)

        pca_features = np.concatenate(pca_features, axis=1)
        norm_features = np.concatenate(norm_features, axis=1)
        norm_features_foreground = np.concatenate(norm_features_foreground,
                                                  axis=1)

    raw_images = np.concatenate(transformed_batch.cpu().numpy().transpose(
        0, 2, 3, 1),
                                axis=1) * 0.23 + 0.4

    channel_variances = np.var(norm_features.reshape(-1,
                                                     norm_features.shape[-1]),
                               axis=0)
    dominant_channel = np.argmax(channel_variances)

    # Visualize the dominant PCA channel as a heatmap
    dominant_map = norm_features[..., dominant_channel]

    fig, axs = plt.subplots(1, 4, figsize=(6, 10))

    axs[0].imshow(raw_images)
    axs[0].set_title('Raw Images')
    axs[0].axis('off')

    axs[1].imshow(norm_features)
    axs[1].set_title('Normalized Features (RGB PCA projection)')
    axs[1].axis('off')

    axs[2].imshow(dominant_map, cmap='viridis')
    axs[2].set_title(
        f'Dominant PCA Channel #{dominant_channel} (Variance = {channel_variances[dominant_channel]:.4f})'
    )
    axs[2].axis('off')

    axs[3].imshow(np.abs(dominant_map), cmap='hot')
    axs[3].set_title('Feature Magnitude (|Dominant PCA Channel|)')
    axs[3].axis('off')

    plt.tight_layout()
    plt.show()
    plt.close()


visualize_dino_features(image[None], batch_pca=False)
