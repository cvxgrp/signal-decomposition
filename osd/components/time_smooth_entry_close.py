''' Time-Smooth, Entry-Close Component (2)

This module contains the class for described a vector valued signal in
 \reals^{T \times p}. This signal is second-order smooth along each column, and
 the rows are penalized for having large variance (entries are close). The cost
 function is

 ϕ(x) = λ_1 Σ_i || D x_i ||_2^2 + λ_2 Σ_t || x_t - μ_t ||_2^2

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
            if key in kwargs.keys() and kwargs[key] is not None:
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
        self.sqrt_P = None
        return

    @property
    def is_convex(self):
        return True

    def _get_cost(self):
        def costfunc(x):
            T, p = x.shape
            if self.sqrt_P is None:
                self.P, self.sqrt_P = make_tsec_mat(T, p,
                                       lambda1=self.lambda1,
                                       lambda2=self.lambda2)
            # P = self.P
            M = self.sqrt_P
            x_flat = x.flatten()
            mu = cvx.Variable(T)
            if isinstance(x, np.ndarray):
                mu.value = np.average(x, axis=1)
            elif isinstance(x, cvx.Variable) and x.value is not None:
                mu.value = np.average(x.value, axis=1)
            x_tilde = cvx.hstack([x_flat, mu])
            # P_param = cvx.Parameter(P.shape, PSD=True, value=P)
            # cost = 0.5 * cvx.quad_form(x_tilde, P)
            cost = 0.5 * cvx.sum_squares(np.sqrt(2) * M @ x_tilde)
            # print(cost.sign, cost.curvature)
            return cost
        return costfunc


    def prox_op(self, v, weight, rho):
        T, p = v.shape
        if self.P is None:
            self.P, self.sqrt_P = make_tsec_mat(T, p, lambda1=self.lambda1,
                                   lambda2=self.lambda2)
        if self.is_constrained:
            if self.F is None:
                # print('making A')
                A = build_constraint_matrix(T, self.period_T, self.vavg_T,
                                            self.first_val_T)
                A = A.tocoo()
                # print('block diag')
                left_block = sp.block_diag([A] * p, format='coo')
                # print('making F')
                data, i, j = left_block.data, left_block.row, left_block.col
                self.F = sp.coo_matrix(
                    (data, (i, j)),
                    shape=(left_block.shape[0], left_block.shape[1] + T)
                )
                # print('done... F.shape = {}'.format(self.F.shape))
                u = build_constraint_rhs(T, self.period_T, self.vavg_T,
                                            self.first_val_T)
                self.g = np.tile(u, p)
        mu = np.average(v, axis=1)
        v_ravel = v.ravel(order='F')
        v_tilde = np.r_[v_ravel, mu]
        # print(self.P.shape, self.F.shape, self.g.shape, v_tilde.shape)
        out_tilde = super().prox_op(v_tilde, weight, rho)
        out_ravel = out_tilde[:-len(mu)]
        out = out_ravel.reshape(v.shape, order='F')
        return out

class TimeSmoothPeriodicEntryClose(TimeSmoothEntryClose):
    def __init__(self, period, lambda1=1, lambda2=1, circular=True, **kwargs):
        P = None
        # don't allow superclass to set any constraints
        for key in ['vavg', 'period', 'first_val']:
            if key in kwargs.keys() and kwargs[key] is not None:
                setattr(self, key + '_' +'T', kwargs[key])
                del kwargs[key]
        super().__init__(lambda1=lambda1, lambda2=lambda2, **kwargs)
        self.sqrt_P = None
        self.period_T = period
        self.circular = circular
        self._internal_constraints = [
            lambda x, T, p: x[period:, :] == x[:-period, :]
        ]
        return

    def _get_cost(self):
        def costfunc(x):
            T, p = x.shape
            q = self.period_T
            if self.sqrt_P is None:
                self.P, self.sqrt_P = make_tsec_mat(q, p,
                                       lambda1=self.lambda1,
                                       lambda2=self.lambda2,
                                       circular=self.circular)
            M = self.sqrt_P
            z = x[:q, :]
            z_flat = z.flatten()
            mu = cvx.Variable(q)
            if isinstance(x, np.ndarray):
                mu.value = np.average(x[:q], axis=1)
            elif isinstance(x, cvx.Variable) and x.value is not None:
                mu.value = np.average(x.value[:q], axis=1)
            z_tilde = cvx.hstack([z_flat, mu])
            # P_param = cvx.Parameter(P.shape, PSD=True, value=P)
            # cost = 0.5 * cvx.quad_form(x_tilde, P)
            cost = 0.5 * cvx.sum_squares(np.sqrt(2) * M @ z_tilde)
            # print(cost.sign, cost.curvature)
            return cost
        return costfunc

    def prox_op(self, v, weight, rho):
        q = self.period_T
        T, p = v.shape
        if self.P is None:
            self.P, self.sqrt_P = make_tsec_mat(q, p, lambda1=self.lambda1,
                                                lambda2=self.lambda2,
                                                circular=self.circular)
        num_groups = T // q
        if T % q != 0:
            num_groups += 1
            num_new_rows = q - T % q
            v_temp = np.r_[v, np.nan * np.ones((num_new_rows, p))]
        else:
            v_temp = v
        v_wrapped = v_temp.reshape((num_groups, q, p))
        v_bar = np.nanmean(v_wrapped, axis=0)
        out_bar = super().prox_op(v_bar, weight, rho)
        out = np.tile(out_bar, (num_groups, 1))
        out = out[:v.shape[0]]
        return out



def make_tsec_mat(T, p, lambda1=1, lambda2=1, circular=False):
    # upper left
    if not circular:
        m1 = sp.eye(m=T - 2, n=T, k=0, format='csr')
        m2 = sp.eye(m=T - 2, n=T, k=1, format='csr')
        m3 = sp.eye(m=T - 2, n=T, k=2, format='csr')
        D = m1 - 2 * m2 + m3
    else:
        m1 = sp.eye(m=T, n=T, k=0, format='csr')
        m2 = sp.eye(m=T, n=T, k=1, format='csr')
        m3 = sp.eye(m=T, n=T, k=2, format='csr')
        m4 = sp.eye(m=T, n=T, k=1-T, format='csr')
        m5 = sp.eye(m=T, n=T, k=2-T, format='csr')
        D = (m1 - 2 * m2 + m3) - 2 * m4 + m5
    upper_left_block = np.sqrt(lambda1) * sp.block_diag([D] * p)
    upper_left_block = upper_left_block.tocoo()
    # upper right
    upper_right_block = None
    # lower left
    data = np.sqrt(lambda2) * np.ones(T * p)
    i = np.arange(T * p)
    j = j_ix_LL(i, T, p)
    lower_left_block =  sp.coo_matrix((data, (i, j)))
    # lower right
    i = np.arange(T * p)
    j = j_ix_LR(i, T, p)
    data = np.sqrt(lambda2) * -1 * np.ones(T * p)
    lower_right_block = sp.coo_matrix((data, (i, j)))
    M = sp.bmat([
        [upper_left_block, upper_right_block],
        [lower_left_block, lower_right_block]
    ])
    return 2 * M.T @ M, M


def j_ix_LL(i, T, p):
    group = i // p
    g_ix = i % p
    j = g_ix * T + group
    return j

def j_ix_LR(i, T, p):
    j = i // p
    return j