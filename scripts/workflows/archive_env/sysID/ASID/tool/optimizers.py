import numpy as np


class Parameter():
    """
    This class implements function to manage parameters, such as learning rate.
    It also allows to have a single parameter for each state of state-action
    tuple.

    """

    def __init__(self, value, min_value=None, max_value=None, size=(1, )):
        """
        Constructor.

        Args:
            value (float): initial value of the parameter;
            min_value (float, None): minimum value that the parameter can reach when decreasing;
            max_value (float, None): maximum value that the parameter can reach when increasing;
            size (tuple, (1,)): shape of the matrix of parameters; this shape can be used to have a single parameter for
                each state or state-action tuple.

        """
        self._initial_value = value
        self._min_value = min_value
        self._max_value = max_value
        self._n_updates = Table(size)

    def __call__(self, *idx, **kwargs):
        """
        Update and return the parameter in the provided index.

        Args:
             *idx (list): index of the parameter to return.

        Returns:
            The updated parameter in the provided index.

        """
        if self._n_updates.table.size == 1:
            idx = list()

        self.update(*idx, **kwargs)

        return self.get_value(*idx, **kwargs)

    def get_value(self, *idx, **kwargs):
        """
        Return the current value of the parameter in the provided index.

        Args:
            *idx (list): index of the parameter to return.

        Returns:
            The current value of the parameter in the provided index.

        """
        new_value = self._compute(*idx, **kwargs)

        if self._min_value is None and self._max_value is None:
            return new_value
        else:
            return np.clip(new_value, self._min_value, self._max_value)

    def _compute(self, *idx, **kwargs):
        """
        Returns:
            The value of the parameter in the provided index.

        """
        return self._initial_value

    def update(self, *idx, **kwargs):
        """
        Updates the number of visit of the parameter in the provided index.

        Args:
            *idx (list): index of the parameter whose number of visits has to be updated.

        """
        self._n_updates[idx] += 1

    @property
    def shape(self):
        """
        Returns:
            The shape of the table of parameters.

        """
        return self._n_updates.table.shape

    @property
    def initial_value(self):
        """
        Returns:
            The initial value of the parameters.

        """
        return self._initial_value


class Optimizer():
    """
    Base class for gradient optimizers.
    These objects take the current parameters and the gradient estimate to compute the new parameters.

    """

    def __init__(self, lr=0.001, maximize=True, *params):
        """
        Constructor

        Args:
            lr ([float, Parameter]): the learning rate;
            maximize (bool, True): by default Optimizers do a gradient ascent step. Set to False for gradient descent.

        """
        if isinstance(lr, float):
            self._lr = Parameter(lr)
        else:
            self._lr = lr
        self._maximize = maximize

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class AdaptiveOptimizer():
    """
    This class implements an adaptive gradient step optimizer.
    Instead of moving of a step proportional to the gradient,
    takes a step limited by a given metric M.
    To specify the metric, the natural gradient has to be provided. If natural
    gradient is not provided, the identity matrix is used.

    The step rule is:

    .. math::
        \\Delta\\theta=\\underset{\\Delta\\vartheta}{argmax}\\Delta\\vartheta^{t}\\nabla_{\\theta}J

        s.t.:\\Delta\\vartheta^{T}M\\Delta\\vartheta\\leq\\varepsilon

    Lecture notes, Neumann G.
    http://www.ias.informatik.tu-darmstadt.de/uploads/Geri/lecture-notes-constraint.pdf

    """

    def __init__(self, eps, maximize=True):
        """
        Constructor.

        Args:
            eps (float): the maximum step defined by the metric;
            maximize (bool, True): by default Optimizers do a gradient ascent step. Set to False for gradient descent.

        """

        self._maximize = maximize
        self._eps = eps

    def __call__(self, params, *args, **kwargs):
        # If two args are passed
        # args[0] is the gradient g, and grads[1] is the natural gradient M^{-1}g
        grads = args[0]
        if len(args) == 2:
            grads = args[1]
        lr = self.get_value(*args, **kwargs)
        if not self._maximize:
            grads *= -1
        return params + lr * grads

    def get_value(self, *args, **kwargs):
        if len(args) == 2:
            gradient = args[0]
            nat_gradient = args[1]
            tmp = (gradient.dot(nat_gradient)).item()
            lambda_v = np.sqrt(tmp / (4. * self._eps))
            # For numerical stability
            lambda_v = max(lambda_v, 1e-8)
            step_length = 1. / (2. * lambda_v)

            return step_length
        elif len(args) == 1:
            return self.get_value(args[0], args[0], **kwargs)
        else:
            raise ValueError('Adaptive parameters needs gradient or gradient'
                             'and natural gradient')


class SGDOptimizer():
    """
    This class implements the SGD optimizer.

    """

    def __init__(self, lr=0.001, maximize=True):
        """
        Constructor.

        Args:
            lr ([float, Parameter], 0.001): the learning rate;
            maximize (bool, True): by default Optimizers do a gradient ascent step. Set to False for gradient descent.

        """
        super().__init__(lr, maximize)

    def __call__(self, params, grads):
        if not self._maximize:
            grads *= -1
        return params + self._lr() * grads


class AdamOptimizer():
    """
    This class implements the Adam optimizer.

    """

    def __init__(self,
                 lr=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 eps=1e-7,
                 maximize=True):
        """
        Constructor.

        Args:
            lr ([float, Parameter], 0.001): the learning rate;
            beta1 (float, 0.9): Adam beta1 parameter;
            beta2 (float, 0.999): Adam beta2 parameter;
            maximize (bool, True): by default Optimizers do a gradient ascent step. Set to False for gradient descent.

        """
        super().__init__(lr, maximize)
        # lr_scheduler must be set to None, as we have our own scheduler
        self._m = None
        self._v = None
        self._beta1 = beta1
        self._beta2 = beta2
        self._eps = eps
        self._t = 0

        self._add_save_attr(_m='numpy',
                            _v='numpy',
                            _beta1='primitive',
                            _beta2='primitive',
                            _t='primitive')

    def __call__(self, params, grads):
        if not self._maximize:
            grads *= -1

        if self._m is None:
            self._t = 0
            self._m = np.zeros_like(params)
            self._v = np.zeros_like(params)

        self._t += 1
        self._m = self._beta1 * self._m + (1 - self._beta1) * grads
        self._v = self._beta2 * self._v + (1 - self._beta2) * grads**2

        m_hat = self._m / (1 - self._beta1**self._t)
        v_hat = self._v / (1 - self._beta2**self._t)

        update = self._lr() * m_hat / (np.sqrt(v_hat) + self._eps)

        return params + update
