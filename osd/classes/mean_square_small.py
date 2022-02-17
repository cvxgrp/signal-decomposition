''' Gaussian Noise Component

This module contains the class for Gaussian Noise

Author: Bennet Meyers
'''

import cvxpy as cvx
import numpy as np
from osd.classes.component import Component

class MeanSquareSmall(Component):

    def __init__(self, size=1, **kwargs):
        super().__init__(**kwargs)
        self.size = size
        return

    @property
    def is_convex(self):
        return True

    def _get_cost(self):
        f = lambda x: cvx.sum_squares(x) / self.size
        return f

    def prox_op(self, v, weight, rho, use_set=None, prox_weights=None):
        a = (2 * weight) / (rho * self.size)
        if prox_weights is not None:
            a /= prox_weights
        r = 1 / (1 + a)
        out = r * np.asarray(v)
        if use_set is not None:
            out[~use_set] = 0
        return out