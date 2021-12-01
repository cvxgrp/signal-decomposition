# -*- coding: utf-8 -*-
''' Smooth Component (1)

This module contains the class for the convex heuristic for a piecewise linear
function. A piecewise constant function has a sparse second-order difference;
many changes in slope are exactly zero and a small number of them can be large.

A convex approximation of this problem is minimizing the L1-norm of the second-
order difference:

    minimize || D_2 x ||_1

This is an extension of the concept of Total Variation filtering, applied to the
differences of a discrete signal, rather than the values.

Author: Bennet Meyers
'''

import cvxpy as cvx
import osqp
import scipy.sparse as sp
from functools import partial
import numpy as np
from osd.components.component import Component
from osd.utilities import compose

class SparseSecondDiffConvex(Component):

    def __init__(self, internal_scale=1., **kwargs):
        super().__init__(**kwargs)
        self._prox_prob = None
        self._rho_over_lambda = None
        self.internal_scale = internal_scale
        self._it = 0
        return

    @property
    def is_convex(self):
        return True

    def _get_cost(self):
        diff2 = partial(cvx.diff, k=2)
        cost = compose(cvx.sum, cvx.abs, lambda x: self.internal_scale * x, diff2)
        return cost

    def prox_op(self, vec_in, weight_val, rho_val,
                verbose=False):
        # print(weight_val)
        problem = self._prox_prob
        ic = self.internal_scale
        rol = rho_val / (weight_val * ic)
        if problem is None:
            P, q, A, l, u = make_all(vec_in, rol)
            problem = osqp.OSQP()
            problem.setup(P=P, q=q, A=A, l=l, u=u, verbose=verbose,
                          eps_rel=1e-4, eps_abs=1e-4)
            self._rho_over_lambda = rol
            self._prox_prob = problem
        else:
            l_new, u_new = make_lu(vec_in, len(vec_in))
            problem.update(l=l_new, u=u_new)
            eps = max(
                (self._it / 100) * 1e-3 + (1 - self._it / 100) * 1e-7,
                1e-9
            )
            if eps >= 1e-5:
                polish = True
            else:
                polish = False
            print('{:.2e}'.format(eps), polish)
            problem.update_settings(eps_abs=eps, eps_rel=eps, polish=polish)
            if ~np.isclose(rol, self._rho_over_lambda, atol=1e-3):
                P_new = make_P(len(vec_in), rol)
                problem.update(Px=P_new)
                self._rho_over_lambda = rol
        results = problem.solve()
        self._it += 0
        return results.x[:len(vec_in)]


def make_P(len_x, rho_over_lambda):
    len_r = len_x - 2
    len_z = len_x
    data = np.ones(len_z) * rho_over_lambda
    i = np.arange(len_z) + len_x + len_r
    P = sp.coo_matrix((data, (i, i)), shape=2 * (len_x + len_r + len_z,))
    return P.tocsc()


def make_q(len_x):
    len_r = len_x - 2
    len_z = len_x
    return np.r_[np.zeros(len_x), np.ones(len_r), np.zeros(len_z)]


def make_A(len_x):
    len_r = len_x - 2
    len_z = len_x
    # block 00
    n = len_x
    m1 = sp.eye(m=n - 2, n=n, k=0)
    m2 = sp.eye(m=n - 2, n=n, k=1)
    m3 = sp.eye(m=n - 2, n=n, k=2)
    B00 = m1 - 2 * m2 + m3
    # block 01
    B01 = sp.eye(len_r)
    # block 10
    B10 = -1 * B00
    # block 11
    B11 = sp.eye(len_r)
    # block 20
    B20 = sp.eye(len_x)
    # block 22
    B22 = -1 * sp.eye(len_z)
    A = sp.bmat([
        [B00, B01, None],
        [B10, B11, None],
        [B20, None, B22]
    ])
    return A.tocsc()


def make_lu(v, len_x):
    len_r = len_x - 2
    len_z = len_x
    l = np.r_[np.zeros(len_r + len_r), v]
    u = np.r_[np.inf * np.ones(len_r + len_r), v]
    return l, u


def make_all(v, rho_over_lambda):
    len_x = len(v)
    P = make_P(len_x, rho_over_lambda)
    q = make_q(len_x)
    A = make_A(len_x)
    l, u = make_lu(v, len_x)
    return P, q, A, l, u