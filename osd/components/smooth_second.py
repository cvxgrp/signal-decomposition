# -*- coding: utf-8 -*-
''' Smooth Component (2)

This module contains the class for a smooth signal that is penalized for large
second-order differences

Author: Bennet Meyers
'''

import scipy.sparse as sp
import numpy as np
import cvxpy as cvx
from functools import partial
from osd.components.component import Component
from osd.utilities import compose
from osd.components.quadlin_utilities import (
    build_constraint_matrix,
    build_constraint_rhs
)

class SmoothSecondDifference(Component):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._c = None
        self._u = None
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
        u = self._u
        cond1 = c is None
        cond2 = self._last_weight != weight
        cond3 = self._last_rho != rho
        if cond1 or cond2 or cond3:
            # print('factorizing the matrix...')
            n = len(v)
            m1 = sp.eye(m=n-2, n=n, k=0)
            m2 = sp.eye(m=n-2, n=n, k=1)
            m3 = sp.eye(m=n-2, n=n, k=2)
            D = m1 - 2 * m2 + m3
            P = 2 * D.T.dot(D) * weight
            M = P + rho * sp.identity(P.shape[0])
            # Build constraints matrix
            A = build_constraint_matrix(
                n, self.period, self.vavg, self.first_val
            )
            if A is not None:
                M = sp.bmat([
                    [M, A.T],
                    [A, None]
                ])
            M = M.tocsc()
            c = sp.linalg.factorized(M)
            u = build_constraint_rhs(
                len(v), self.period, self.vavg, self.first_val
            )
            self._c = c
            self._u = u
            self._last_weight = weight
            self._last_rho = rho
        if u is not None:
            rhs = np.r_[rho * v, u]
            out = c(rhs)
            out = out[:len(v)]
        else:
            rhs = rho * v
            out = c(rhs)
        super().prox_op(v, weight, rho)
        return out
