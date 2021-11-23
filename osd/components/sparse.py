# -*- coding: utf-8 -*-
''' Laplace Noise Component

This module contains the class for Laplace noise, or a noise term modeled
as a random variable drawn from a Laplace distribution. The Laplace distribution
has a tighter peak and fatter tails than a Gaussian distribution, and so is a
good model for a signal that is often zero and sometime quite large. For this
reason, it is often used as a heuristic for sparsity.

The cost function for Laplace noise is simply the sum of the absolute values,
or the L1 norm.

Author: Bennet Meyers
'''

import cvxpy as cvx
import osqp
import scipy.sparse as sp
from osd.components.component import Component
from osd.utilities import compose
import numpy as np

class Sparse(Component):

    def __init__(self,chunk_size=None, **kwargs):
        super().__init__(**kwargs)
        self.chunk_size = chunk_size
        if chunk_size is not None:
            self._prox_prob = None
            self._rho_over_lambda = None
            self._it = 0
            self.internal_scale = 1
            def make_const(x, T, p):
                nc = (T - 1) // chunk_size + 1
                z = cvx.Variable(nc)
                A = np.zeros((nc, T))
                for i in range(nc):
                    A[
                        i, i * chunk_size:(i + 1) * chunk_size
                    ] = np.ones(chunk_size)
                return A.T @ z == x
            self._internal_constraints = [make_const]
        return

    @property
    def is_convex(self):
        return True

    def _get_cost(self):
        cost = compose(cvx.sum, cvx.abs)
        return cost

    def prox_op(self, v, weight, rho, verbose=False):
        if self.chunk_size is None:
            kappa = weight / rho
            t1 = v - kappa
            t2 = -v - kappa
            x = np.clip(t1, 0, np.inf) - np.clip(t2, 0, np.inf)
            return x
        else:
            problem = self._prox_prob
            ic = self.internal_scale
            rol = rho / (weight * ic * self.chunk_size)
            if problem is None:
                P, q, A, l, u = make_all(v, self.chunk_size, rol)
                problem = osqp.OSQP()
                problem.setup(P=P, q=q, A=A, l=l, u=u, verbose=verbose,
                              eps_abs=1e-4, eps_rel=1e-4)
                self._rho_over_lambda = rol
                self._prox_prob = problem
            else:
                l_new, u_new = make_lu(v, len(v), self.chunk_size)
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
                    P_new = make_P(len(v), self.chunk_size, rol)
                    problem.update(Px=P_new)
                    self._rho_over_lambda = rol
            results = problem.solve()
            self._it += 0
            return results.x[:len(v)]
            # return results
            # num_chunks = (len(v) - 1) // self.chunk_size + 1
            # return results.x[len(v)+num_chunks:2*len(v)+num_chunks]



def make_P(len_x, chunk_size, rho_over_lambda):
    len_r = (len_x - 1) // chunk_size + 1
    len_z = len_x
    len_s = (len_x - 1) // chunk_size + 1
    data = np.ones(len_z) * rho_over_lambda
    i = np.arange(len_z) + len_x + len_r
    P = sp.coo_matrix((data, (i, i)), shape=2 * (len_x + len_r + len_z + len_s,))
    return P.tocsc()


def make_q(len_x, chunk_size):
    len_r = (len_x - 1) // chunk_size + 1
    len_z = len_x
    len_s = (len_x - 1) // chunk_size + 1
    return np.r_[np.zeros(len_x), np.ones(len_r), np.zeros(len_z),
                 np.zeros(len_s)]


def make_A(len_x, chunk_size):
    len_r = (len_x - 1) // chunk_size + 1
    len_z = len_x
    len_s = (len_x - 1) // chunk_size + 1
    # block 01
    B01 = sp.eye(len_r)
    # block 03
    B03 = sp.eye(len_s)
    if not len_x % chunk_size == 0:
        remainder = len_x % chunk_size
        rs = remainder / chunk_size
        B03.data[-1][-1] = rs
    # block 11
    B11 = sp.eye(len_r)
    # block 13
    B13 = -1 * B03
    # block 20
    B20 = sp.eye(len_x)
    # block 22
    B22 = 1 * sp.eye(len_z)
    # block 30
    B30 = -1 * np.eye(len_x)
    # block 33
    data = np.ones(len_x)
    i = np.arange(len_x)
    j = i // chunk_size
    # print(i, j, len_s, len_x)
    B33 = sp.coo_matrix((data, (i, j)), shape=(len_x, len_s))

    A = sp.bmat([
        [None, B01, None, B03],
        [None, B11, None, B13],
        [B20, None, B22, None],
        [B30, None, None, B33]
    ])
    return A.tocsc()


def make_lu(v, len_x, chunk_size):
    len_r = (len_x - 1) // chunk_size + 1
    len_z = len_x
    len_s = (len_x - 1) // chunk_size + 1
    l = np.r_[np.zeros(len_r + len_r), v, np.zeros(len_x)]
    u = np.r_[np.inf * np.ones(len_r + len_r), v, np.zeros(len_x)]
    return l, u


def make_all(v, chunk_size, rho_over_lambda):
    len_x = len(v)
    P = make_P(len_x, chunk_size, rho_over_lambda)
    q = make_q(len_x, chunk_size)
    A = make_A(len_x, chunk_size)
    l, u = make_lu(v, len_x, chunk_size)
    return P, q, A, l, u