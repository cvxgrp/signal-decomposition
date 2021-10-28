''' Time-Smooth, Entry-Close Component (2)

This module contains the class for described a vector valued signal in
 \reals^{T \times p}. This signal is second-order smooth along each column, and
 the rows are penalized for having large variance (entries are close). The cost
 function is

 phi(x) = λ_1 Σ_i || D x_i ||_2^2 + λ_2 Σ_t || x_t - μ_t ||_2^2

 where D is the second-order difference matrix and μ \in \reals^T is an
 internal variable

Author: Bennet Meyers
'''

import scipy.sparse as sp
import numpy as np
import cvxpy as cvx
from osd.components import QuadLin
from osd.components.quadlin_utilities import (
    build_constraint_matrix,
    build_constraint_rhs
)

class TimeSmoothEntryClose(QuadLin):

    def __init__(self, lambda1=1, lambda2=1, **kwargs):
        self.is_constrained = False
        for key in ['vavg', 'period', 'first_val']:
            if key in kwargs.keys():
                setattr(self, key + '_' +'T', kwargs[key])
                del kwargs[key]
                self.is_constrained = True
            else:
                setattr(self, key + '_' + 'T', None)
        P = None
        q = None
        r = None
        F = None
        g = None
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        if self.is_constrained:
            self._internal_constraints = []
            if self.vavg_T is not None:
                self._internal_constraints.append(
                    lambda x, T, p: cvx.sum(x, axis=0) / T == self.vavg_T
                )
            if self.period_T is not None:
                per = self.period_T
                self._internal_constraints.append(
                    lambda x, T, p: x[per:, :] == x[:-per, :]
                )
            if self.first_val_T is not None:
                self._internal_constraints.append(
                    lambda x, T, p: x[0, :] == self.first_val_T
                )
        super().__init__(P, q=q, r=r, F=F, g=g, **kwargs)
        return

    @property
    def is_convex(self):
        return True

    def _get_cost(self):
        def costfunc(x):
            T, p = x.shape
            if self.P is None:
                self.P = make_tsec_mat(T, p,
                                       lambda1=self.lambda1,
                                       lambda2=self.lambda2)
            P = self.P
            x_flat = x.flatten()
            mu = cvx.Variable(T)
            if isinstance(x, np.ndarray):
                mu.value = np.average(x, axis=1)
            elif isinstance(x, cvx.Variable) and x.value is not None:
                mu.value = np.average(x.value, axis=1)
            x_tilde = cvx.hstack([x_flat, mu])
            cost = 0.5 * cvx.quad_form(x_tilde, P)
            return cost
        return costfunc


    def prox_op(self, v, weight, rho):
        T, p = v.shape
        if self.P is None:
            self.P = make_tsec_mat(T, p, lambda1=self.lambda1,
                                   lambda2=self.lambda2)
        if self.is_constrained:
            if self.F is None:
                A = build_constraint_matrix(T, self.vavg_T, self.period_T,
                                            self.first_val_T)
                self.F = sp.bmat([
                    [sp.block_diag([A] * p), None],
                    [None, np.zeros((T, T))]
                ])
                u = build_constraint_rhs(T, self.vavg_T, self.period_T,
                                            self.first_val_T)
                self.g = np.r_[np.tile(u, p), np.zeros(T)]
        mu = np.average(v, axis=1)
        v_ravel = v.ravel(order='F')
        v_tilde = np.r_[v_ravel, mu]
        out_tilde = super().prox_op(v_tilde, weight, rho)
        out_ravel = out_tilde[:-len(mu)]
        out = out_ravel.reshape(v.shape, order='F')
        return out


def make_tsec_mat(T, p, lambda1=1, lambda2=1):
    # upper left
    m1 = sp.eye(m=T - 2, n=T, k=0)
    m2 = sp.eye(m=T - 2, n=T, k=1)
    m3 = sp.eye(m=T - 2, n=T, k=2)
    D = m1 - 2 * m2 + m3
    upper_left_block = np.sqrt(lambda1) * sp.block_diag([D] * p)
    # upper right
    upper_right_block = None
    # lower left
    data = np.ones(T * p)
    i = np.arange(T * p)
    j = j_ix(i, T, p)
    lower_left_block = np.sqrt(lambda2) * sp.coo_matrix((data, (i, j)))
    # lower right
    a = -1 * np.ones(p).reshape((p, -1))
    lower_right_block = np.sqrt(lambda2) * sp.block_diag([a] * T)
    M = sp.bmat([
        [upper_left_block, upper_right_block],
        [lower_left_block, lower_right_block]
    ])
    return 2 * M.T @ M


def j_ix(i, T, p):
    group = i // p
    g_ix = i % p
    j = g_ix * T + group
    return j