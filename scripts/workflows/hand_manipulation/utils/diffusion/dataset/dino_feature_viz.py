from scripts.workflows.hand_manipulation.utils.diffusion.backbone.dino_feature import DinoV2Encoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
import torchvision.transforms as T
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from scripts.workflows.hand_manipulation.utils.diffusion.dataset.utils import ColorRandomizer, GaussianNoiseRandomizer

dino_encoder = DinoV2Encoder('dinov2_vits14', "x_norm_patchtokens")

transform = T.Compose([
    T.Resize((1120, 1120), interpolation=T.InterpolationMode.BICUBIC),
    # T.CenterCrop(560),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


def get_dino_feature(obs_data, dino_kwargs):

    if dino_kwargs is None:
        return obs_data

    transform = T.Compose([
        T.Resize((dino_kwargs["resize_shape"]),
                 interpolation=T.InterpolationMode.BICUBIC),
        # T.CenterCrop(560),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    B = obs_data.shape[0]

    pca = PCA(n_components=dino_kwargs["pca_n_components"])
    scaler = MinMaxScaler()

    batch_size = 64
    pca_features = []

    if dino_kwargs.get("add_randomizer", False):

        color_randomizer = ColorRandomizer(obs_data.shape[1:])
        gaussian_randomizer = GaussianNoiseRandomizer(obs_data.shape[1:], )

    with torch.no_grad():
        for i in range(0, B, batch_size):
            transformed_images = []
            sub_batch_data = obs_data[i:i + batch_size]
            for img_np in sub_batch_data:

                if dino_kwargs.get("add_randomizer", False):

                    img_np = (gaussian_randomizer.forward(img_np / 255) *
                              255).astype(np.uint8)

                    img_np = np.asarray(color_randomizer.get_transform()(
                        Image.fromarray(
                            img_np.transpose(1, 2,
                                             0).astype(np.uint8)))).transpose(
                                                 2, 0, 1)

                # Convert (3, H, W) to (H, W, 3) if needed
                if img_np.shape[0] == 3 and img_np.shape[-1] != 3:
                    img_np = np.transpose(img_np, (1, 2, 0))

                # Convert to uint8 PIL Image
                img_pil = Image.fromarray((img_np).astype(np.uint8))

                # Apply DINOv2 transform
                img_tensor = transform(img_pil)
                transformed_images.append(img_tensor)
            transformed_batch = torch.stack(transformed_images).to(
                dino_encoder.device)  # (B, 3, H, W)

            batch = transformed_batch[i:i +
                                      batch_size]  # shape: (<=64, 3, H, W)
            feat = dino_encoder(batch)  # shape: (b, N_patch, D)
            #         features_list.append(feat)

            # features = torch.cat(features_list, dim=0)  # shape: (B, N_patch, D)
            num_data, N_patch, D = feat.shape

            if not dino_kwargs.get("batch_pca", True):

                for i in range(num_data):
                    feature = feat[i].cpu().numpy()

                    pca.fit(feature)
                    pca_feature = pca.transform(feature)
                    norm_feature = scaler.fit_transform(pca_feature)

                    dim = int(np.sqrt(norm_feature.shape[0]))

                    # plt.imshow(norm_feature.reshape(dim, dim, -1))
                    # plt.show()
                    pca_features.append(
                        norm_feature.reshape(dim, dim, -1)[None])

            else:
                if len(feat) == 0:
                    continue

                num_data, n_patch, dim = feat.shape
                features = feat.reshape(num_data * n_patch, dim)
                features = features.cpu()

                pca.fit(features)

                features_foreground = pca.transform(features)
                norm_feature = scaler.fit_transform(features_foreground)
                dim = int(np.sqrt(norm_feature.shape[0] / batch_size))
                pca_features.append(
                    norm_feature.reshape(batch_size, dim, dim, -1))

                # pca_features = np.concatenate(pca_features, axis=1)

        pca_features = np.concatenate(pca_features, axis=0)

    return pca_features.transpose(0, 3, 1, 2)  # (B, C, H, W)


def visualize_dino_features(
    raw_images,
    threshold=0.9,
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
    # fig, axs = plt.subplots(1, batch_size)
    # if batch_size == 1:
    #     axs = [axs]
    # for i in range(batch_size):
    #     img_t = (transformed_batch[i] * 0.23 + 0.4)
    #     axs[i].imshow(img_t.permute(1, 2, 0).cpu().numpy())
    #     axs[i].axis('off')  # Turn off axis labels

    # plt.show()
    # plt.close()

    # Extract DINO features
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

    transformed_batch
    raw_images = np.concatenate(transformed_batch.cpu().numpy().transpose(
        0, 2, 3, 1),
                                axis=1) * 0.23 + 0.4

    fig, axs = plt.subplots(2, 1)
    axs[0].imshow(raw_images)
    axs[0].set_title('Raw Images')

    axs[1].imshow(norm_features)
    axs[1].set_title('Normalized Features')

    # axs[2].imshow(norm_features)
    # axs[2].set_title('Normalized Features')
    # axs[3].imshow(norm_features_foreground)
    # axs[3].set_title('Foreground Features')

    axs[0].axis('off')
    axs[1].axis('off')
    # axs[2].axis('off')
    # axs[3].axis('off')

    plt.tight_layout()
    plt.show()
    plt.close()


# image = np.array(Image.open('/home/ensu/Downloads/image_1.png').convert('RGB'))
# visualize_dino_features(image[None], batch_pca=False)
