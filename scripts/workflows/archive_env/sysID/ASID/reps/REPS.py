import numpy as np

from scipy.optimize import minimize

from scripts.workflows.sysID.ASID.tool.gaussian import GaussianDiagonalDistribution
import torch


class REPS():
    """
    Episodic Relative Entropy Policy Search algorithm.
    "A Survey on Policy Search for Robotics", Deisenroth M. P., Neumann G.,
    Peters J.. 2013.

    """

    def __init__(self,
                 dynamics,
                 running_cost,
                 horizon=1,
                 nu=1,
                 eps=1.0,
                 init_mean=0.5,
                 init_cov_diag=0.5,
                 kappa=None):
        """
        Constructor.

        Args:
            eps ([float, Parameter]): the maximum admissible value for the Kullback-Leibler
                divergence between the new distribution and the
                previous one at each update step.

        """
        self._eps = eps
        self.T = horizon
        self.nu = nu
        self.init_mean = init_mean
        self.init_cov_diag = init_cov_diag
        self.F = dynamics
        self.running_cost = running_cost
        self.kappa = kappa

    def reset(self):

        self.mean = np.zeros(self.T * self.nu) + self.init_mean
        self.cov = np.ones(self.T * self.nu) * self.init_cov_diag
        self.distribution = GaussianDiagonalDistribution(self.mean, self.cov)

    def evaluate_trajetories(self, min_batch_size=32):

        target_pc_seq = self.F.target_buffer["color_pc"][0, ..., :3]
        train_pc_seq = self.F.train_buffer["color_pc"][..., :3]

        from scripts.workflows.sysID.ASID.tool.utilis import evaluate_chamfer_distance
        loss = evaluate_chamfer_distance(target_pc_seq, train_pc_seq)

        return loss

    def _update(self):

        Jep = self.evaluate_trajetories()

        Jep = Jep / np.max(Jep)  # normalize rewards

        theta = self.F.train_buffer["deform_physical_params"][:,
                                                              0].cpu().numpy()
        eta_start = np.ones(1)

        res = minimize(REPS._dual_function,
                       eta_start,
                       jac=REPS._dual_function_diff,
                       bounds=((np.finfo(np.float32).eps, np.inf), ),
                       args=(self._eps, Jep, theta))

        eta_opt = res.x.item()

        Jep -= np.max(Jep)

        dd = np.exp(Jep / eta_opt)

        if self.kappa is not None:
            self.distribution.con_wmle(theta, dd, self._eps, self.kappa)

        else:
            self.distribution.mle(theta, dd)

    @staticmethod
    def _dual_function(eta_array, *args):
        eta = eta_array.item()
        eps, Jep, theta = args

        max_J = np.max(Jep)

        re = Jep - max_J
        sum1 = np.mean(np.exp(re / eta))

        return eta * eps + eta * np.log(sum1) + max_J

    @staticmethod
    def _dual_function_diff(eta_array, *args):
        eta = eta_array.item()
        eps, Jep, theta = args

        max_J = np.max(Jep)

        r = Jep - max_J

        sum1 = np.mean(np.exp(r / eta))
        sum2 = np.mean(np.exp(r / eta) * r)

        gradient = eps + np.log(sum1) - sum2 / (eta * sum1)

        return np.array([gradient])
