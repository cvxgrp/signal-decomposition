# -*- coding: utf-8 -*-
''' Asymmetric Noise Component

This module contains the class noise generated from an asymmetric Laplace
distribution (ALD): https://en.wikipedia.org/wiki/Asymmetric_Laplace_distribution

The cost function is the "tilted L1" or "quantile regression" cost function,
which is is just a modified absolute value function. This parameter tau is
between zero and one, and it set the approximate quantile of the residual
distribution that the model is fit to.

    See: https://colab.research.google.com/github/cvxgrp/cvx_short_course/blob/master/applications/quantile_regression.ipynb

When tau is set to 0.5, quantile regression becomes "median regression" and is
equivalent to using an L1 cost function, which is appropriate for Laplace noise.

In contrast, the traditional sum-of-squares or L2 cost function fits the approximate
average of the residual distribution, which is appropriate for Gaussian noise.


Author: Bennet Meyers
'''

import cvxpy as cvx
import numpy as np
from osd.classes.component import Component
from osd.utilities import compose

class AsymmetricNoise(Component):

    def __init__(self, tau=0.85, **kwargs):
        super().__init__(tau=tau, **kwargs)
        self._tau  = tau
        return

    @property
    def is_convex(self):
        return True

    def _get_cost(self):
        tau = self.parameters['tau']
        quant = lambda x: 0.5 * cvx.abs(x) + (tau - 0.5) * x
        cost = compose(cvx.sum, quant)
        return cost

    def _get_params(self):
        tau = cvx.Parameter(nonneg=True)
        return {'tau': tau}

    def prox_op(self, v, weight, rho, use_set=None, prox_weights=None):
        kappa = weight / (2 * rho)
        tau = self._tau
        a = v + (weight / rho) * (0.5 - tau)
        x = np.clip(a - kappa, 0, np.inf) - np.clip(-a - kappa, 0, np.inf)
        if use_set is not None:
            x[~use_set] = 0
        return x