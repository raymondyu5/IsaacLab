import torch
from torch.distributions.multivariate_normal import MultivariateNormal


class OnlineCEM:

    def __init__(self,
                 dim_latent,
                 num_samples=64,
                 num_elite=10,
                 u_min=-1.0,
                 u_max=1.0,
                 init_mean=0.0,
                 init_cov_diag=0.5,
                 device="cpu"):
        self.device = device
        self.dim_latent = dim_latent
        self.num_samples = num_samples
        self.num_elite = num_elite

        self.mean = torch.full((dim_latent, ), init_mean, device=device)
        self.cov = torch.eye(dim_latent, device=device) * init_cov_diag
        self.cov_reg = torch.eye(dim_latent,
                                 device=device) * init_cov_diag * 1e-6

        self.u_min = torch.full((dim_latent, ), u_min, device=device)
        self.u_max = torch.full((dim_latent, ), u_max, device=device)

    def sample_candidates(self):
        """Sample candidate latent noise vectors for evaluation."""

        eps = 1e-3  # or larger
        cov = self.cov + self.cov_reg + eps * torch.eye(self.cov.shape[0],
                                                        device=self.cov.device)
        dist = MultivariateNormal(self.mean, cov)
        samples = dist.sample((self.num_samples, ))
        return torch.clamp(samples, self.u_min, self.u_max)

    def update_distribution(self, samples, costs):
        """Update mean/cov using elites (lowest-cost samples)."""

        num_elite = min(self.num_elite, len(samples))
        top_costs, top_idx = torch.topk(costs.reshape(-1),
                                        num_elite,
                                        largest=True)
        elites = samples[top_idx]

        self.mean = elites.mean(dim=0)
        xm = elites - self.mean
        self.cov = (xm.T @ xm) / num_elite + self.cov_reg

        return elites[0], top_costs[0]  # best sample + its cost


def evaluate_episode(noise):
    """
    Replace with your rollout logic.
    Noise stays fixed for the entire episode.
    Return scalar cost (lower is better).
    """
    target = torch.ones_like(noise) * 0.3
    return torch.norm(noise - target).item()


# # Online usage
# cem = OnlineCEM(dim_latent=4, num_samples=32, num_elite=5)

# for episode in range(500):
#     # 1. Sample candidates for this round
#     samples = cem.sample_candidates()

#     # 2. Run each candidate once (episodes or parallel envs)
#     costs = torch.tensor([evaluate_episode(s) for s in samples])

#     # 3. Update distribution with the results
#     best_noise, best_cost = cem.update_distribution(samples, costs)

#     print(
#         f"[Episode {episode}] Best cost: {best_cost:.4f}, mean: {cem.mean.cpu().numpy()}"
#     )
