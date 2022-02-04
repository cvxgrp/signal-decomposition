# -*- coding: utf-8 -*-
''' Linear Trend Component

This module contains the class for a linear trend with respect to time

Author: Bennet Meyers
'''

import cvxpy as cvx
import numpy as np
from osd.classes.component import Component


class LinearTrend(Component):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.z = cvx.Variable(2)
        self._internal_constraints = [
            lambda x, T, p: cvx.diff(x, k=2) == 0
        ]
        return

    @property
    def is_convex(self):
        return True

    def _get_cost(self):
        cost = lambda x: 0
        return cost

    def prox_op(self, v, weight, rho, use_set=None, prox_weights=None):
        T = len(v)
        A = np.c_[np.ones(T), np.arange(T)]
        if use_set is not None:
            A_tilde = A[use_set, :]
            v_tilde = v[use_set]
        else:
            A_tilde = A
            v_tilde = v
        if prox_weights is not None:
            pw = prox_weights[use_set] if use_set is not None else prox_weights
            A_tilde = A_tilde.T
            A_tilde *= pw
            A_tilde = A_tilde.T
            v_tilde *= pw
        if self.first_val is not None:
            c = np.zeros_like(v)
            c[0] = 1
            C = c.reshape((1, -1)) @ A
            temp_mat = np.block([
                [2 * A_tilde.T @ A_tilde, C.T],
                [C, np.atleast_2d([0])]
            ])
            v_tilde = np.r_[2 * A_tilde.T @ v_tilde, [self.first_val]]
            A_tilde = temp_mat
        x, _, _, _ = np.linalg.lstsq(A_tilde, v_tilde, rcond=None)
        out = A @ x[:2]
        return out
