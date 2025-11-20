import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderMLP(nn.Module):

    def __init__(self, input_dim, latent_dim, hidden_dims, device="cuda"):
        super().__init__()
        self.device = device

        layers = []
        last_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, h_dim))
            layers.append(nn.ReLU())
            last_dim = h_dim

        self.model = nn.Sequential(*layers).to(device)
        self.fc_mu = nn.Linear(last_dim, latent_dim).to(device)
        self.fc_logvar = nn.Linear(last_dim, latent_dim).to(device)

    def forward(self, state, action):
        h = self.model(action)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class DecoderMLP(nn.Module):

    def __init__(self, latent_dim, output_dim, hidden_dims, device="cuda"):
        super().__init__()
        self.device = device

        layers = []
        last_dim = latent_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, h_dim))
            layers.append(nn.ReLU())
            last_dim = h_dim

        layers.append(nn.Linear(last_dim, output_dim))
        self.model = nn.Sequential(*layers).to(device)

    def forward(self, state, z):
        return self.model(z), None


class DiscriminatorMLP(nn.Module):

    def __init__(self, latent_dim, device="cuda"):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(),
                                   nn.Linear(64, 32), nn.ReLU(),
                                   nn.Linear(32, 2)).to(device)

    def forward(self, z):
        return self.model(z)


class FactorVAE(nn.Module):

    def __init__(self, input_dim, latent_dim, hidden_dims, gamma=1, lr=1e-4):
        super().__init__()
        self.latent_dim = latent_dim
        self.gamma = gamma
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.encoder = EncoderMLP(input_dim, latent_dim, hidden_dims,
                                  self.device)
        self.decoder = DecoderMLP(latent_dim, input_dim, hidden_dims[::-1],
                                  self.device)
        self.discriminator = DiscriminatorMLP(latent_dim, self.device)
        self.D_z_reserve = None  # Used for discriminator update
        self.vae_optimizer = torch.optim.Adam(list(self.encoder.parameters()) +
                                              list(self.decoder.parameters()),
                                              lr=lr)
        self.disc_optimizer = torch.optim.Adam(self.discriminator.parameters(),
                                               lr=lr)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def permute_latent(self, z):
        # Permute each dimension across the batch
        B, D = z.size()
        permuted = []
        for d in range(D):
            permuted.append(z[:, d][torch.randperm(B)])
        return torch.stack(permuted, dim=1)

    def forward(self, state, action):
        mu, logvar = self.encoder(state, action)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(state, z)[0]
        return recon, z, action, mu, logvar

    def loss(self, recon, input, mu, logvar, z, optimizer_idx, kld_weight):
        if optimizer_idx == 0:
            recon_loss = F.mse_loss(recon, input, reduction='mean')
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),
                                        dim=1).mean()
            D_z = self.discriminator(z)
            vae_tc_loss = (D_z[:, 0] - D_z[:, 1]).mean()
            return recon_loss + kld_weight * kld_loss + self.gamma * vae_tc_loss

        elif optimizer_idx == 1:
            B = input.size(0)
            device = input.device
            true_labels = torch.ones(B, dtype=torch.long, device=device)
            false_labels = torch.zeros(B, dtype=torch.long, device=device)

            z = z.detach()
            z_perm = self.permute_latent(z)
            D_z = self.discriminator(z)
            D_z_perm = self.discriminator(z_perm)

            return 0.5 * (F.cross_entropy(D_z, false_labels) +
                          F.cross_entropy(D_z_perm, true_labels))

    def train_epoches(self, train_loader, val_loader, epoches, update_ratio=1):
        epoch_train_losses = []
        epoch_val_losses = []

        for epoch in range(epoches):
            train_loss_sum = 0
            val_loss_sum = 0

            for state, action in train_loader:
                state = state.to(self.device)
                action = action.to(self.device)

                # === Forward pass ===
                recon, z, input, mu, logvar = self(state, action)

                # === VAE update ===
                self.vae_optimizer.zero_grad()
                vae_loss = self.loss(recon,
                                     input,
                                     mu,
                                     logvar,
                                     z,
                                     optimizer_idx=0,
                                     kld_weight=1.0)

                vae_loss.backward()
                self.vae_optimizer.step()

                # === Discriminator update (multiple times) ===
                for _ in range(update_ratio):
                    self.disc_optimizer.zero_grad()

                    # ❗ Recompute z for discriminator without gradients
                    with torch.no_grad():
                        mu, logvar = self.encoder(state, action)
                        z_disc = self.reparameterize(mu, logvar)

                    # 🔥 Only pass z_disc, and avoid any reuse of graph-connected tensors
                    disc_loss = self.loss(None,
                                          action,
                                          None,
                                          None,
                                          z_disc,
                                          optimizer_idx=1,
                                          kld_weight=1.0)
                    disc_loss.backward()
                    self.disc_optimizer.step()

                train_loss_sum += vae_loss.item()

            epoch_train_losses.append(train_loss_sum / len(train_loader))

            # === Validation ===

            val_loss_sum = 0  # ✅ reset at beginning of validation
            with torch.no_grad():
                for state, action in val_loader:
                    state = state.to(self.device)
                    action = action.to(self.device)
                    recon, z, input, mu, logvar = self(state, action)
                    val_loss = self.loss(recon,
                                         input,
                                         mu,
                                         logvar,
                                         z,
                                         optimizer_idx=0,
                                         kld_weight=1.0)

                    val_loss_sum += val_loss.item()
                epoch_val_losses.append(val_loss_sum / len(val_loader))

            print(f"Epoch {epoch}: Train Loss: {epoch_train_losses[-1]:.4f}, "
                  f"Val Loss: {epoch_val_losses[-1]:.4f}")
