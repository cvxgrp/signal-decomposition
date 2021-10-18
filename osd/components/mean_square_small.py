# -*- coding: utf-8 -*-
''' Gaussian Noise Component

This module contains the class for Gaussian Noise

Author: Bennet Meyers
'''

import cvxpy as cvx
import numpy as np
from osd.components.component import Component

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

    def prox_op(self, v, weight, rho):
        a = (2 * weight) / (rho * self.size)
        r = 1 / (1 + a)
        return r * np.asarray(v)