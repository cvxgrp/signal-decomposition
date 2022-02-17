# -*- coding: utf-8 -*-
''' Smooth Component (1)

This module contains the class for a smooth signal that is penalized for large
first-order differences

Author: Bennet Meyers
'''

import scipy.sparse as sp
import cvxpy as cvx
from functools import partial
from osd.classes.quad_lin import QuadLin
from osd.utilities import compose
from osd.classes.quadlin_utilities import (
    build_constraint_matrix,
    build_constraint_rhs
)

class SmoothFirstDifference(QuadLin):

    def __init__(self, **kwargs):
        P = None
        q = None
        r = None
        F = None
        g = None
        super().__init__(P, q=q, r=r, F=F, g=g, **kwargs)
        return

    @property
    def is_convex(self):
        return True

    def _get_cost(self):
        diff1 = partial(cvx.diff, k=1)
        cost = compose(cvx.sum_squares, diff1)
        return cost

    def prox_op(self, v, weight, rho, use_set=None, prox_weights=None):
        n = len(v)
        if self.P is None:
            m1 = sp.eye(m=n - 1, n=n, k=0)
            m2 = sp.eye(m=n - 1, n=n, k=1)
            D = m2 - m1
            self.P = 2 * D.T.dot(D)
            self.F = build_constraint_matrix(
                n, self.period, self.vavg, self.first_val
            )
            if self.F is not None:
                self.g = build_constraint_rhs(
                    len(v), self.period, self.vavg, self.first_val
                )
        vout = super().prox_op(v, weight, rho, use_set=use_set)
        return vout