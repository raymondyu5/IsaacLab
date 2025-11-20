import numpy as np
import torch


def load_pca_data(path, device="cuda", hand_side="right"):

    eigengrasps_info = np.load(path + f"/{hand_side}_pca.npy",
                               allow_pickle=True).item()
    eigen_vectors = torch.as_tensor(
        eigengrasps_info["eigen_vectors"]).to(device)

    min_values = torch.as_tensor(eigengrasps_info["lower_values"]).to(device)
    max_values = torch.as_tensor(eigengrasps_info["upper_values"]).to(device)
    D_mean = torch.as_tensor(eigengrasps_info["D_mean"]).to(device)
    D_std = torch.as_tensor(eigengrasps_info["D_std"]).to(device)
    # # Expand min/max to the full original space

    # min_orig_norm = torch.matmul(self.min_values, self.eigen_vectors)
    # max_orig_norm = torch.matmul(self.max_values, self.eigen_vectors)
    # # Now pick the true min and max
    # final_min = torch.minimum(min_orig_norm, max_orig_norm)
    # final_max = torch.maximum(min_orig_norm, max_orig_norm)
    # # Step 2: Denormalize
    # self.min_orig = final_min * self.D_std + self.D_mean
    # self.max_orig = final_max * self.D_std + self.D_mean
    return eigen_vectors, min_values, max_values, D_mean, D_std


# -----------------------
# Decode function
# -----------------------
def reconstruct_hand_pose_from_normalized_action(a_normalized, eigen_vectors,
                                                 min_values, max_values,
                                                 D_mean, D_std):
    """
    Args:
        a_normalized: (B, K) or (K,), RL output in [-1, 1]
        eigen_vectors: (K, D)
        min_values: (K,)
        max_values: (K,)
        D_mean: (D,)
        D_std: (D,)
    
    Returns:
        x: (B, D) if batched, or (D,) if single
    """
    # If a_normalized is (K,), unsqueeze to (1, K)
    # single_input = False
    # if a_normalized.dim() == 1:
    #     a_normalized = a_normalized.unsqueeze(0)
    #     single_input = True

    # Step 1: Scale normalized action to eigengrasp range

    if isinstance(a_normalized, np.ndarray):
        # NumPy version
        a_scaled = (a_normalized + 1) / 2 * (max_values -
                                             min_values) + min_values  # (B, K)
        x_norm = a_scaled @ eigen_vectors  # (B, D)
        x = (x_norm * D_std + D_mean)[0]  # (B, D)
    else:
        # PyTorch version
        a_scaled = (a_normalized + 1) / 2 * (max_values -
                                             min_values) + min_values  # (B, K)
        x_norm = torch.matmul(a_scaled, eigen_vectors)  # (B, D)
        x = x_norm * D_std + D_mean  # (B, D)
    return x


def decode_latent_action(a_normalized, eigen_vectors, lower_values,
                         upper_values, D_mean, D_std):
    """
    Args:
        a_normalized: (B, K) or (K,), latent in [-1, 1]
        eigen_vectors: (K, D)
        lower_values, upper_values: (K,)
        D_mean, D_std: (D,)
    Returns:
        x: (B, D) or (D,) in original hand space
    """
    single_input = False
    if a_normalized.dim() == 1:
        a_normalized = a_normalized.unsqueeze(0)
        single_input = True

    # # Step 1: scale latent [-1,1] → [lower, upper]
    # a_scaled = (a_normalized + 1) / 2 * (upper_values -
    #                                      lower_values) + lower_values  # (B, K)
    eig_mean = (upper_values + lower_values) / 2
    eig_std = (upper_values - lower_values) / 2

    a_scaled = eig_mean + a_normalized * eig_std

    # Step 2: back-project
    x_norm = torch.matmul(a_scaled, eigen_vectors)  # (B, D)

    # Step 3: undo normalization
    D_mean = D_mean.view(1, -1)
    D_std = D_std.view(1, -1)
    x = (x_norm * D_std + D_mean)
    hand_actions = (x + torch.pi) % (2 * torch.pi) - torch.pi

    return hand_actions
