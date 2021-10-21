# -*- coding: utf-8 -*-
''' Smooth Component (2)

This module contains the class for a smooth signal that is penalized for large
second-order differences

Author: Bennet Meyers
'''

import scipy.linalg as spl
import scipy.sparse as sp
import numpy as np
import cvxpy as cvx
from functools import partial
from osd.components.component import Component
from osd.utilities import compose

class SmoothSecondDifference(Component):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._c = None
        self._last_weight = None
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

    def prox_op(self, v, weight, rho):
        c = self._c
        cond1 = c is None
        cond2 = self._last_weight != weight
        cond3 = self._last_rho != rho
        if cond1 or cond2 or cond3:
            n = len(v)
            m1 = sp.eye(m=n-2, n=n, k=0)
            m2 = sp.eye(m=n-2, n=n, k=1)
            m3 = sp.eye(m=n-2, n=n, k=2)
            D = m1 - 2 * m2 + m3
            P = 2 * D.T.dot(D) * weight
            M = P + rho * sp.identity(P.shape[0])
            M = M.tocsc()
            c = sp.linalg.factorized(M)
            self._c = c
            # M = np.diff(np.eye(n), axis=0, n=2)
            # r = 2 * weight / rho
            # ab = np.zeros((3, n))
            # A = np.eye(n) + r * M.T.dot(M)
            # for i in range(3):
            #     ab[i] = np.pad(np.diag(A, k=i), (0, i))
            # c = spl.cholesky_banded(ab, lower=True)
            # self._c = c
            # self._last_weight = weight
            # self._last_rho = rho
        return c(rho * v)
