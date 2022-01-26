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
from osd.components.quad_lin import QuadLin
from osd.utilities import compose
from osd.components.quadlin_utilities import (
    build_constraint_matrix,
    build_constraint_rhs
)

class SmoothSecondDifference(QuadLin):

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
        internal_scaling = 1 #10 ** (3.5 / 2)
        diff2 = partial(cvx.diff, k=2)
        cost = compose(lambda x: internal_scaling * x, diff2)
        cost = compose(cvx.sum_squares, cost)
        return cost

    def prox_op(self, v, weight, rho, use_set=None, prox_counts=None):
        n = len(v)
        if self.P is None:
            self.P = make_l2d2matrix(n)
            self.F = build_constraint_matrix(
                n, self.period, self.vavg, self.first_val
            )
            if self.F is not None:
                self.g = build_constraint_rhs(
                    len(v), self.period, self.vavg, self.first_val
                )
        vout = super().prox_op(v, weight, rho, use_set=use_set)
        return vout

class SmoothSecondDiffPeriodic(SmoothSecondDifference):
    def __init__(self, period, circular=True, **kwargs):
        for key in ['vavg', 'period', 'first_val']:
            if key in kwargs.keys() and kwargs[key] is not None:
                setattr(self, key + '_' +'T', kwargs[key])
                del kwargs[key]
        super().__init__(**kwargs)
        self.period_T = period
        self.circular = circular
        self._internal_constraints = [
            lambda x, T, p: x[period:, :] == x[:-period, :]
        ]
        return

    def prox_op(self, v, weight, rho, use_set=None, prox_counts=None):
        n = len(v)
        q = self.period_T
        num_groups = n // q
        if use_set is not None:
            v_tilde = np.copy(v)
            v_tilde[use_set] = np.nan
        else:
            v_tilde = v
        if n % q != 0:
            num_groups += 1
            num_new_rows = q - n % q
            v_temp = np.r_[v_tilde, np.nan * np.ones(num_new_rows)]
        else:
            v_temp = v
        v_wrapped = v_temp.reshape((num_groups, q))
        v_bar = np.nanmean(v_wrapped, axis=0)
        if self.P is None:
            self.P = make_l2d2matrix(q, circular=self.circular)
            # full diff matrix is T-2, but circular diff matrix is q
            self.P *= (1 - 2 / (q * num_groups))
            self.F = build_constraint_matrix(
                q, None, self.vavg, self.first_val
            )
            if self.F is not None:
                self.g = build_constraint_rhs(
                    q, None, self.vavg, self.first_val
                )
        out_bar = super().prox_op(v_bar, weight, rho)
        out = np.tile(out_bar, num_groups)
        out = out[:n]
        return out


def make_l2d2matrix(n, circular=False):
    if not circular:
        m1 = sp.eye(m=n - 2, n=n, k=0, format='csr')
        m2 = sp.eye(m=n - 2, n=n, k=1, format='csr')
        m3 = sp.eye(m=n - 2, n=n, k=2, format='csr')
        D = m1 - 2 * m2 + m3
    else:
        m1 = sp.eye(m=n, n=n, k=0, format='csr')
        m2 = sp.eye(m=n, n=n, k=1, format='csr')
        m3 = sp.eye(m=n, n=n, k=2, format='csr')
        m4 = sp.eye(m=n, n=n, k=1-n, format='csr')
        m5 = sp.eye(m=n, n=n, k=2-n, format='csr')
        D = (m1 - 2 * m2 + m3) - 2 * m4 + m5
    return 2 * D.T.dot(D)