# -*- coding: utf-8 -*-
''' Smooth Component (1)

This module contains the class for a smooth signal that is penalized for large
first-order differences

Author: Bennet Meyers
'''

import scipy.linalg as spl
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

class SmoothFirstDifference(Component):

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
        diff1 = partial(cvx.diff, k=1)
        cost = compose(cvx.sum_squares, diff1)
        return cost

    def prox_op(self, v, weight, rho):
        c = self._c
        u = self._u
        cond1 = c is None
        cond2 = self._last_weight != weight
        cond3 = self._last_rho != rho
        if cond1 or cond2 or cond3:
            n = len(v)
            m1 = sp.eye(m=n-1, n=n, k=0)
            m2 = sp.eye(m=n-1, n=n, k=1)
            D = m2 - m1
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
            self._c = c
            u = build_constraint_rhs(
                len(v), self.period, self.vavg, self.first_val
            )
            self._u = u
        if u is not None:
            rhs = np.r_[rho * v, u]
            out = c(rhs)
            out = out[:len(v)]
        else:
            rhs = rho * v
            out = c(rhs)
        return out