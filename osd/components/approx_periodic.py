# -*- coding: utf-8 -*-
''' Approximatelly Periodic (1)

This module contains the class for a signal that nearly repeats with period p

Author: Bennet Meyers
'''

import scipy.linalg as spl
import numpy as np
import cvxpy as cvx
from osd.components.component import Component
from osd.utilities import compose

class ApproxPeriodic(Component):

    def __init__(self, period, **kwargs):
        self._approx_period = period
        self._internal_constraints = [
            lambda x, T, K: cvx.sum(x[:period]) == 0,
            lambda x, T, K: cvx.sum(x[-period:]) == 0,
            lambda x, T, K: cvx.sum(x) == 0
        ]
        super().__init__(**kwargs)
        self._c = None
        self._last_weight = None
        self._last_rho = None
        return

    @property
    def is_convex(self):
        return True

    def _get_cost(self):
        p = self._approx_period
        diff_p = lambda x: x[p:] - x[:-p]
        cost = compose(cvx.sum_squares, diff_p)
        return cost

    def prox_op(self, v, weight, rho):
        c = self._c
        cond1 = c is None
        cond2 = self._last_weight != weight
        cond3 = self._last_rho != rho
        if cond1 or cond2 or cond3:
            n = len(v)
            I = np.eye(n)
            p = self._approx_period
            M = I[p:] - I[:-p]
            r = 2 * weight / rho
            ab = np.zeros((2, n))
            A = np.eye(n) + r * M.T.dot(M)
            for i in range(2):
                ab[i] = np.pad(np.diag(A, k=i), (0, i))
            c = spl.cholesky_banded(ab, lower=True)
            self._c = c
        return spl.cho_solve_banded((c, True), v)