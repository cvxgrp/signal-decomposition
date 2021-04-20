# -*- coding: utf-8 -*-
''' Laplace Noise Component

This module contains the class for Laplace noise, or a noise term modeled
as a random variable drawn from a Laplace distribution. The Laplace distribution
has a tighter peak and fatter tails than a Gaussian distribution, and so is a
good model for a signal that is often zero and sometime quite large. For this
reason, it is often used as a heuristic for sparsity.

The cost function for Laplace noise is simply the sum of the absolute values,
or the L1 norm.

Author: Bennet Meyers
'''

import cvxpy as cvx
from osd.components.component import Component
from osd.utilities import compose
import numpy as np

class LaplaceNoise(Component):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    @property
    def is_convex(self):
        return True

    def _get_cost(self):
        cost = compose(cvx.sum, cvx.abs)
        return cost

    def prox_op(self, v, theta, rho):
        kappa = theta / rho
        t1 = v - kappa
        t2 = -v - kappa
        x = np.clip(t1, 0, np.inf) - np.clip(t2, 0, np.inf)
        return x