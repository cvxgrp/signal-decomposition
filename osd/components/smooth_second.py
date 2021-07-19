# -*- coding: utf-8 -*-
''' Smooth Component (2)

This module contains the class for a smooth signal that is penalized for large
second-order differences

Author: Bennet Meyers
'''

import scipy.linalg as spl
import numpy as np
import cvxpy as cvx
from functools import partial
from osd.components.component import Component
from osd.utilities import compose

class SmoothSecondDifference(Component):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._c = None
        self._last_theta = None
        self._last_rho = None
        return

    @property
    def is_convex(self):
        return True

    def _get_cost(self):
        internal_scaling = 1 #10 ** (3.5 / 2)
        diff2 = partial(cvx.diff, k=2)
        cost = compose(lambda x: internal_scaling * x, diff2)
        cost = compose(cvx.sum_squares, cost)
        return cost

    def prox_op(self, v, theta, rho):
        c = self._c
        cond1 = c is None
        cond2 = self._last_theta != theta
        cond3 = self._last_rho != rho
        if cond1 or cond2 or cond3:
            n = len(v)
            M = np.diff(np.eye(n), axis=0, n=2)
            r = 2 * theta / rho
            ab = np.zeros((3, n))
            A = np.eye(n) + r * M.T.dot(M)
            for i in range(3):
                ab[i] = np.pad(np.diag(A, k=i), (0, i))
            c = spl.cholesky_banded(ab, lower=True)
            self._c = c
            self._last_theta = theta
            self._last_rho = rho
        return spl.cho_solve_banded((c, True), v)
