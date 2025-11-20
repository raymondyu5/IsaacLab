import torch
import time
import logging
from torch.distributions.multivariate_normal import MultivariateNormal

logger = logging.getLogger(__name__)


def pytorch_cov(x, rowvar=False, bias=False, ddof=None, aweights=None):
    """Estimates covariance matrix like numpy.cov"""
    # ensure at least 2D
    if x.dim() == 1:
        x = x.view(-1, 1)

    # treat each column as a data point, each row as a variable
    if rowvar and x.shape[0] != 1:
        x = x.t()

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    w = aweights
    if w is not None:
        if not torch.is_tensor(w):
            w = torch.tensor(w, dtype=torch.float)
        w_sum = torch.sum(w)
        avg = torch.sum(x * (w / w_sum)[:, None], 0)
    else:
        avg = torch.mean(x, 0)

    # Determine the normalization
    if w is None:
        fact = x.shape[0] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * torch.sum(w * w) / w_sum

    xm = x.sub(avg.expand_as(x))

    if w is None:
        X_T = xm.t()
    else:
        X_T = torch.mm(torch.diag(w), xm).t()

    conv = torch.mm(X_T, xm)
    conv = conv / fact

    return conv


class CEM():
    """
    Cross Entropy Method control
    This implementation batch samples the trajectories and so scales well with the number of samples K.
    """

    def __init__(self,
                 dynamics,
                 running_cost,
                 nx,
                 nu,
                 num_samples=100,
                 num_iterations=3,
                 num_elite=30,
                 horizon=15,
                 device="cpu",
                 terminal_state_cost=None,
                 u_min=None,
                 u_max=None,
                 choose_best=False,
                 init_cov_diag=0.5,
                 init_mean=0.5):
        """

        :param dynamics: function(state, action) -> next_state (K x nx) taking in batch state (K x nx) and action (K x nu)
        :param running_cost: function(state, action) -> cost (K x 1) taking in batch state and action (same as dynamics)
        :param nx: state dimension
        :param num_samples: K, number of trajectories to sample
        :param horizon: T, length of each trajectory
        :param device: pytorch device
        :param terminal_state_cost: function(state) -> cost (K x 1) taking in batch state
        """
        self.d = device
        self.dtype = torch.double  # TODO determine dtype
        self.K = num_samples  # N_SAMPLES
        self.T = horizon  # TIMESTEPS
        self.M = num_iterations
        self.num_elite = num_elite
        self.choose_best = choose_best

        # dimensions of state and control
        self.nx = nx
        self.nu = nu

        self.mean = None
        self.cov = None

        self.F = dynamics
        self.running_cost = running_cost
        self.terminal_state_cost = terminal_state_cost
        self.init_cov_diag = init_cov_diag
        self.init_mean = init_mean
        self.u_min = u_min
        self.u_max = u_max
        # make sure if any of them is specified, both are specified
        if self.u_max is not None and self.u_min is None:
            self.u_min = -self.u_max
        if self.u_min is not None and self.u_max is None:
            self.u_max = -self.u_min
        if self.u_min is not None:
            self.u_min = self.u_min.to(device=self.d)
            self.u_max = self.u_max.to(device=self.d)
        self.action_distribution = None

        # regularize covariance
        self.cov_reg = torch.eye(self.T * self.nu,
                                 device=self.d,
                                 dtype=self.dtype) * init_cov_diag * 1e-5

    def reset(self):
        """
        Clear controller state after finishing a trial
        """
        # action distribution, initialized as N(0,I)
        # we do Hp x 1 instead of H x p because covariance will be Hp x Hp matrix instead of some higher dim tensor
        self.mean = torch.zeros(
            self.T * self.nu, device=self.d, dtype=self.dtype) + self.init_mean
        self.cov = torch.eye(self.T * self.nu, device=self.d,
                             dtype=self.dtype) * self.init_cov_diag

    def sample_parameter(self, num_envs):

        # in case it's singular
        self.action_distribution = MultivariateNormal(
            self.mean, covariance_matrix=torch.clamp(self.cov, 1e-4, 1000))
        print("sample range", self.mean, self.cov)

        samples_parameter = torch.clamp(
            self.action_distribution.sample((num_envs, )),
            self.u_min.to(self.d), self.u_max.to(self.d))
        return samples_parameter

    def evaluate_trajetories(self, min_batch_size):

        target_pc_seq = self.F.target_buffer["seg_pc"].squeeze(1).to("cuda")[
            ..., :3]
        train_pc_seq = self.F.train_buffer["seg_pc"].to("cuda")[..., :3]

        from scripts.workflows.sysID.ASID.tool.utilis import evaluate_chamfer_distance
        reward, self.loss = evaluate_chamfer_distance(target_pc_seq,
                                                      train_pc_seq)
        del target_pc_seq, train_pc_seq
        torch.cuda.empty_cache()

        top_costs, topk = torch.topk(self.loss,
                                     min(self.num_elite, len(self.loss)),
                                     largest=False,
                                     sorted=False)

        return top_costs, topk

    def command(self, min_batch_size=32):

        top_costs, topk = self.evaluate_trajetories(min_batch_size)
        target_physical_propoerties = self.F.target_buffer[
            "deform_physical_params"][0][0]

        train_physical_propoerties = self.F.train_buffer[
            "deform_physical_params"][:, 0, 0]

        self.mean = torch.mean(train_physical_propoerties[topk], dim=0)

        self.cov = pytorch_cov(train_physical_propoerties[topk], rowvar=False)
