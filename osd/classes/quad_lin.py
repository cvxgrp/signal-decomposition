''' Quad-linear component

This module contains the class for a signal described by a quadradic cost
restricted to an affine set, of the form

phi(x) = (1/2) x^T P X + q^T x + r
dom f = {x | F x = g}

Author: Bennet Meyers
'''

import scipy.sparse as sp
import numpy as np
import cvxpy as cvx
from osd.classes.component import Component
from osd.masking import (
    make_masked_identity_matrix,
    make_mask_matrix,
    make_inverse_mask_matrix
)

class QuadLin(Component):

    def __init__(self, P, q=None, r=None, F=None, g=None, **kwargs):
        super().__init__(**kwargs)
        self.P = P
        self.q = q
        self.r = r
        self.F = F
        if g is None and F is not None:
            self.g = np.zeros(F.shape[0])
        else:
            self.g = g
        self.prox_M = None
        self.prox_MtM = None
        self.prox_Mt = None
        self._c = None
        self._u = None
        self._last_weight = None
        self._last_rho = None
        if F is not None:
            self._internal_constraints = [
                lambda x, T, p: F @ x == self.g
            ]
        return

    @property
    def is_convex(self):
        return True

    def _get_cost(self):
        def costfunc(x):
            cost = 0.5 * cvx.quad_form(x, self.P)
            if self.q is not None:
                cost += self.q.T @ x
            if self.r is not None:
                cost += self.r
            return cost
        return costfunc

    def prox_op(self, v, weight, rho, use_set=None, prox_weights=None):
        c = self._c
        u = self._u
        # cached problem does not exist
        cond1 = c is None
        # check if the sparsity pattern has changed. if so, don't use cached
        # problem
        if not cond1 and use_set is not None:
            test_mat = make_mask_matrix(use_set)
            # first test if shape has changed
            if self.prox_M.shape != test_mat.shape:
                cond1 = True
            # assuming the shape has not changed, check if entries are the same
            elif (self.prox_M != make_mask_matrix(use_set)).nnz > 0:
                cond1 = True
        cond2 = self._last_weight != weight
        cond3 = self._last_rho != rho
        if use_set is not None and cond1:
            self.prox_M = make_mask_matrix(use_set)
            self.prox_Mt = make_inverse_mask_matrix(use_set)
            self.prox_MtM = make_masked_identity_matrix(use_set)
            if prox_weights is not None:
                self.prox_MtM.data *= prox_weights[prox_weights != 0]
        if cond1 or cond2 or cond3:
            # print('factorizing the matrix...')
            n = len(v)
            if self.prox_MtM is None:
                temp_mat = sp.identity(self.P.shape[0])
            else:
                temp_mat = self.prox_MtM
            M = weight * self.P + rho * temp_mat
            if self.F is not None:
                A = sp.csc_matrix(self.F)
                M = sp.bmat([
                    [M, A.T],
                    [A, None]
                ])
            M = M.tocsc()
            # print('factorizing matrix of size ({} x {}) with {} nnz'.format(
            #     *M.shape, M.nnz
            # ))
            c = sp.linalg.factorized(M)
            # print('done factorizing!')
            if self.F is not None:
                u = self.g
            self._c = c
            self._u = u
            self._last_weight = weight
            self._last_rho = rho
        if use_set is None:
            if self.q is None:
                upper = rho * v
            else:
                upper = rho * v - weight * self.q
        else:
            if self.q is None:
                upper = rho * self.prox_MtM @ v
            else:
                upper = rho * self.prox_MtM @ v - weight * self.q
        if u is not None:
            rhs = np.r_[upper, u]
            # print(rhs.shape)
            out = c(rhs)
            out = out[:len(v)]
        else:
            rhs = upper
            out = c(rhs)
        super().prox_op(v, weight, rho, use_set=use_set)
        return out