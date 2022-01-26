# -*- coding: utf-8 -*-
''' Linear Trend Component

This module contains the class for a linear trend with respect to time

Author: Bennet Meyers
'''

import cvxpy as cvx
import numpy as np
from osd.components.component import Component


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

    def prox_op(self, v, weight, rho, use_set=None, prox_counts=None):
        T = len(v)
        A = np.c_[np.ones(T), np.arange(T)]
        if use_set is not None:
            A_tilde = A[use_set, :]
            v_tilde = v[use_set]
        else:
            A_tilde = A
            v_tilde = v
        x, _, _, _ = np.linalg.lstsq(A_tilde, v_tilde, rcond=None)
        out = A @ x
        return out
