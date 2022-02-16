# -*- coding: utf-8 -*-
''' Approximatelly Periodic (1)

This module contains the class for a signal that nearly repeats with period p

Author: Bennet Meyers
'''

import scipy.sparse as sp
import cvxpy as cvx
from osd.classes.quad_lin import QuadLin
from osd.utilities import compose
from osd.classes.quadlin_utilities import (
    build_constraint_matrix,
    build_constraint_rhs
)

class ApproxPeriodic(QuadLin):

    def __init__(self, period, **kwargs):
        self._approx_period = period
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
        p = self._approx_period
        diff_p = lambda x: x[p:] - x[:-p]
        cost = compose(cvx.sum_squares, diff_p)
        return cost

    def prox_op(self, v, weight, rho, use_set=None, prox_weights=None):
        n = len(v)
        per = self._approx_period
        if self.P is None:
            m1 = sp.eye(m=n - per, n=n, k=0)
            m2 = sp.eye(m=n - per, n=n, k=per)
            D = m2 - m1
            self.P = 2 * D.T.dot(D)
            self.F = build_constraint_matrix(
                n, None, self.vavg, self.first_val
            )
            if self.F is not None:
                self.g = build_constraint_rhs(
                    len(v), None, self.vavg, self.first_val
                )
        vout = super().prox_op(v, weight, rho, use_set=use_set)
        return vout