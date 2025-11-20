import numpy as np

from scipy.optimize import minimize

from scripts.workflows.sysID.ASID.tool.gaussian import GaussianDiagonalDistribution
import torch


def minibatch_number(size, batch_size):
    """
    Function to retrieve the number of batches, given a batch sizes.

    Args:
        size (int): size of the dataset;
        batch_size (int): size of the batches.

    Returns:
        The number of minibatches in the dataset.

    """
    return int(np.ceil(size / batch_size))


def minibatch_generator(batch_size, *dataset):
    """
    Generator that creates a minibatch from the full dataset.

    Args:
        batch_size (int): the maximum size of each minibatch;
        dataset: the dataset to be splitted.

    Returns:
        The current minibatch.

    """
    size = len(dataset[0])
    num_batches = minibatch_number(size, batch_size)
    indexes = np.arange(0, size, 1)
    np.random.shuffle(indexes)
    batches = [(i * batch_size, min(size, (i + 1) * batch_size))
               for i in range(0, num_batches)]

    for (batch_start, batch_end) in batches:
        batch = []
        for i in range(len(dataset)):
            batch.append(dataset[i][indexes[batch_start:batch_end]])
        yield batch


class AbstractGaussianTorchDistribution():
    """
    Gaussian distribution with diagonal covariance matrix. The parameters
    vector represents the mean and the standard deviation for each dimension.

    """

    def __init__(self, context_shape=None):
        """
        Constructor.

        Args:
            context_shape (Tuple): shape of the context variable.

        """

    def distribution(self, context=None):
        mu, chol_sigma = self._get_mean_and_chol(context)
        return torch.distributions.MultivariateNormal(loc=mu,
                                                      scale_tril=chol_sigma,
                                                      validate_args=False)

    def sample(self, context=None, num_env=None):
        dist = self.distribution(context)

        return dist.sample((num_env, ))

    def log_pdf(self, theta, context=None):
        dist = self.distribution(context)
        return dist.log_prob(theta)

    def __call__(self, theta, context=None):
        return torch.exp(self.log_pdf(theta, context))

    def mean(self, context=None):
        mu, _ = self._get_mean_and_chol(context)
        return mu

    def entropy(self, context=None):
        dist = self.distribution(context)
        return dist.entropy()

    def mle(self, theta, weights=None):
        raise NotImplementedError

    def con_wmle(self, theta, weights, eps, kappa):
        raise NotImplementedError

    def diff_log(self, theta, context=None):
        raise NotImplementedError

    def _get_mean_and_chol(self, context):
        raise NotImplementedError

    def parameters(self):
        raise NotImplementedError


# import kaolin.metrics.pointcloud as pc
class DiagonalGaussianTorchDistribution(AbstractGaussianTorchDistribution):

    def __init__(self, mu, sigma, device="cpu"):
        self._mu = torch.nn.Parameter(mu)
        self._log_sigma = torch.nn.Parameter(torch.log(sigma))
        self.device = device

    def get_parameters(self):
        rho = torch.empty(self.parameters_size).to(self.device)
        n_dims = len(self._mu)

        rho[:n_dims] = self._mu
        rho[n_dims:] = self._log_sigma

        return rho

    def set_parameters(self, rho):
        n_dims = len(self._mu)
        self._mu.data = rho[:n_dims]
        self._log_sigma.data = rho[n_dims:]

    @property
    def parameters_size(self):
        return 2 * len(self._mu)

    def parameters(self):
        return [self._mu, self._log_sigma]

    def _get_mean_and_chol(self, context):
        return self._mu, torch.diag(torch.exp(self._log_sigma))


class ePPO():
    """
    Episodic Relative Entropy Policy Search algorithm.
    "A Survey on Policy Search for Robotics", Deisenroth M. P., Neumann G.,
    Peters J.. 2013.

    """

    def __init__(self,
                 dynamics,
                 running_cost,
                 optimizer,
                 horizon=1,
                 nu=1,
                 init_mean=0.5,
                 init_cov_diag=0.5,
                 n_epochs_policy=1,
                 batch_size=32,
                 eps_ppo=2,
                 ent_coeff=2,
                 device="cpu"):
        """
        Constructor.

        Args:
            eps ([float, Parameter]): the maximum admissible value for the Kullback-Leibler
                divergence between the new distribution and the
                previous one at each update step.

        """

        self.T = horizon
        self.nu = nu
        self.init_mean = init_mean
        self.init_cov_diag = init_cov_diag
        self.F = dynamics
        self.running_cost = running_cost
        self.optimizer = optimizer

        self._n_epochs_policy = n_epochs_policy
        self._batch_size = batch_size
        self._eps_ppo = eps_ppo
        self._ent_coeff = ent_coeff
        self.device = device

    def reset(self):

        self.mean = (torch.zeros(self.T * self.nu) + self.init_mean).to(
            self.device)
        self.cov = (torch.ones(self.T * self.nu) * self.init_cov_diag).to(
            self.device)

        self.distribution = DiagonalGaussianTorchDistribution(
            self.mean,
            self.cov,
            device=self.device,
        )
        self._optimizer = self.optimizer['class'](
            self.distribution.parameters(),
            **self.optimizer['params'],
        )

    def evaluate_trajetories(self, min_batch_size=32):

        target_pc_seq = self.F.target_buffer["color_pc"][0, ..., :3]
        train_pc_seq = self.F.train_buffer["color_pc"][..., :3]

        from scripts.workflows.sysID.ASID.tool.utilis import evaluate_chamfer_distance
        loss = evaluate_chamfer_distance(target_pc_seq, train_pc_seq)

        return loss

    def _update(self):

        Jep = self.evaluate_trajetories()

        Jep = Jep / np.max(Jep)  # normalize rewards

        theta = self.F.train_buffer["deform_physical_params"][:, 0].to(
            self.device)
        context = self.F.target_buffer["deform_physical_params"][:, 0].to(
            self.device)

        Jep = torch.tensor(Jep).to(self.device)
        J_mean = torch.mean(Jep).to(self.device)
        J_std = torch.std(Jep).to(self.device)

        Jep = (Jep - J_mean) / (J_std + 1e-8)

        old_dist = self.distribution.log_pdf(theta).detach()

        full_batch = (theta, Jep, old_dist, context)

        for epoch in range(self._n_epochs_policy):
            for minibatch in minibatch_generator(self._batch_size,
                                                 *full_batch):

                theta_i, context_i, Jep_i, old_dist_i = self._unpack(minibatch)

                self._optimizer.zero_grad()
                prob_ratio = torch.exp(
                    self.distribution.log_pdf(theta_i, context_i) - old_dist_i)
                clipped_ratio = torch.clamp(prob_ratio, 1 - self._eps_ppo,
                                            1 + self._eps_ppo)
                loss = -torch.mean(
                    torch.min(prob_ratio * Jep_i, clipped_ratio * Jep_i))
                loss -= self._ent_coeff * self.distribution.entropy(context_i)
                loss.backward()
                self._optimizer.step()

        return loss

    def _unpack(self, minibatch):

        theta_i, Jep_i, old_dist_i, context_i = minibatch

        return theta_i, context_i, Jep_i, old_dist_i
