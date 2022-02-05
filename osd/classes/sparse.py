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
import scipy.sparse as sp
from osd.classes.component import Component
from osd.utilities import compose
import numpy as np
import warnings

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
                    ] = 1
                return A.T @ z == x
            self._internal_constraints = [make_const]
        return

    @property
    def is_convex(self):
        return True

    def _get_cost(self):
        cost = compose(cvx.sum, cvx.abs)
        return cost

    def prox_op(self, v, weight, rho, use_set=None, prox_counts=None):
        if self.chunk_size is None:
            kappa = weight / rho
            t1 = v - kappa
            t2 = -v - kappa
            x = np.clip(t1, 0, np.inf) - np.clip(t2, 0, np.inf)
            if use_set is not None:
                x[~use_set] = 0
        else:
            cs = self.chunk_size
            cn = (len(v) - 1) // cs + 1
            remainder = len(v) % cs

            if use_set is not None:
                v_temp = np.copy(v)
                v_temp[~use_set] = np.nan
            else:
                v_temp = v
            if remainder == 0:
                v_bar = v_temp
            else:
                v_bar = np.r_[v_temp, np.nan * np.ones(cs - remainder)]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                nan_counts = np.sum(np.isnan(v_bar.reshape((cn, cs))), axis=1)
                v_bar = np.nanmean(v_bar.reshape((cn, cs)), axis=1)

            if remainder > 0:
                nan_counts[-1] -= cs - remainder

            if np.any(np.isnan(v_bar)):
                v_bar[np.isnan(v_bar)] = 0

            kappa = np.zeros(cn)
            kappa[nan_counts != cs] =(
                weight / (rho * (1 - nan_counts[nan_counts != cs] / cs))
            )

            # kappa = weight / rho * np.ones(cn)

            t1 = v_bar - kappa
            t2 = -v_bar - kappa

            x = np.tile(np.clip(t1, 0, np.inf) - np.clip(t2, 0, np.inf),
                        (cs, 1)).ravel(order='F')
            x = x[:len(v)]
        return x



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