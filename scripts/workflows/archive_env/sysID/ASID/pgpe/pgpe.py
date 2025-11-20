import numpy as np

from scipy.optimize import minimize

from scripts.workflows.sysID.ASID.tool.gaussian import GaussianDiagonalDistribution
import torch
# import kaolin.metrics.pointcloud as pc


class PGPE():
    """
    Episodic Relative Entropy Policy Search algorithm.
    "A Survey on Policy Search for Robotics", Deisenroth M. P., Neumann G.,
    Peters J.. 2013.

    """

    def __init__(
        self,
        dynamics,
        running_cost,
        optimizer,
        horizon=1,
        nu=1,
        init_mean=0.5,
        init_cov_diag=0.5,
    ):
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
        context = self.F.target_buffer["deform_physical_params"][:, 0].cpu(
        ).numpy()
        baseline_num_list = list()
        baseline_den_list = list()
        diff_log_dist_list = list()

        for i in range(len(Jep)):
            J_i = Jep[i]
            theta_i = theta[i]

            diff_log_dist = self.distribution.diff_log(theta_i, context)
            diff_log_dist2 = diff_log_dist**2

            diff_log_dist_list.append(diff_log_dist)
            baseline_num_list.append(J_i * diff_log_dist2)
            baseline_den_list.append(diff_log_dist2)

        # Compute baseline
        baseline = np.mean(baseline_num_list, axis=0) / np.mean(
            baseline_den_list, axis=0)
        baseline[np.logical_not(np.isfinite(baseline))] = 0.

        # Compute gradient
        grad_J_list = list()
        for i in range(len(Jep)):
            diff_log_dist = diff_log_dist_list[i]
            J_i = Jep[i]

            grad_J_list.append(diff_log_dist * (J_i - baseline))

        grad_J = np.mean(grad_J_list, axis=0)

        omega_old = self.distribution.get_parameters()
        omega_new = self.optimizer(omega_old, grad_J)
        self.distribution.set_parameters(omega_new)

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
