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
from osd.classes.quad_lin import QuadLin
from osd.utilities import compose
from osd.classes.quadlin_utilities import (
    build_constraint_matrix,
    build_constraint_rhs
)
from osd.classes.base_graph_class import GraphComponent

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

    def prox_op(self, v, weight, rho, use_set=None, prox_weights=None):
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
        vout = super().prox_op(v, weight, rho, use_set=use_set,
                               prox_weights=prox_weights)
        return vout

    def make_graph_form(self, T, p):
        gf = SmoothSecondDifferenceGraph(
            self.weight, T, p,
            vmin=self.vmin, vmax=self.vmax,
            period=self.period, first_val=self.first_val
        )
        self._gf = gf
        return gf.make_dict()

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
            lambda x, T, p: x[period:] == x[:-period]
        ]
        return

    def prox_op(self, v, weight, rho, use_set=None, prox_weights=None):
        n = len(v)
        q = self.period_T
        num_groups = n // q
        if use_set is None:
            use_set = ~np.isnan(v)
        else:
            use_set = np.logical_and(~np.isnan(v), use_set)
        if use_set is not None:
            v_tilde = np.copy(v)
            v_tilde[use_set] = np.nan
        else:
            v_tilde = v
        if n % q != 0:
            num_groups += 1
            num_new_rows = q - n % q
            v_temp = np.r_[v_tilde, np.nan * np.ones(num_new_rows)]
            u_temp = np.r_[use_set, np.zeros(num_new_rows, dtype=bool)]
        else:
            v_temp = v
            u_temp = use_set
        print(v_temp)
        v_wrapped = v_temp.reshape((num_groups, q))
        u_wrapped = u_temp.reshape((num_groups, q))
        v_bar = np.nanmean(v_wrapped, axis=0)
        u_bar = np.any(u_wrapped, axis=0)
        counts = np.sum(u_wrapped, axis=0)
        prox_weights = counts / num_groups
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
        out_bar = super().prox_op(v_bar, weight, rho, use_set=u_bar,
                                  prox_weights=prox_weights)
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
        D = (m1 - 2 * m2 + m3) - 2 * m4 + m5x
    return 2 * D.T.dot(D)

class SmoothSecondDifferenceGraph(GraphComponent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        return

    def __set_z_size(self):
        self._z_size = self.x_size - 2

    def __make_P(self):
        P1 = sp.dok_matrix(2 * (self.x_size,))
        P2 = np.sqrt(self.weight) * sp.eye(self.z_size)
        self._Px = P1
        self._Pz = P2

    def __make_A(self):
        T = self.x_size
        m1 = sp.eye(m=T - 2, n=T, k=0, format='csr')
        m2 = sp.eye(m=T - 2, n=T, k=1, format='csr')
        m3 = sp.eye(m=T - 2, n=T, k=2, format='csr')
        self._A = m1 - 2 * m2 + m3