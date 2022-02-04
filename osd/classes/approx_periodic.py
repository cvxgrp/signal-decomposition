# -*- coding: utf-8 -*-
''' Approximatelly Periodic (1)

This module contains the class for a signal that nearly repeats with period p

Author: Bennet Meyers
'''
# TODO: convert this to quad-lin
import scipy.linalg as spl
import numpy as np
import cvxpy as cvx
from osd.classes.component import Component
from osd.masking import Mask
from osd.utilities import compose

class ApproxPeriodic(Component):

    def __init__(self, period, **kwargs):
        self._approx_period = period
        self._internal_constraints = [
            lambda x, T, p: cvx.sum(x[:period]) == 0,
            lambda x, T, p: cvx.sum(x[-period:]) == 0,
            lambda x, T, p: cvx.sum(x) == 0
        ]
        super().__init__(**kwargs)
        self._c = None
        self._last_weight = None
        self._last_rho = None
        self._mask = None
        return

    @property
    def is_convex(self):
        return True

    def _get_cost(self):
        p = self._approx_period
        diff_p = lambda x: x[p:] - x[:-p]
        cost = compose(cvx.sum_squares, diff_p)
        return cost

    def prox_op(self, v, weight, rho, use_set=None, prox_weights=None):
        c = self._c
        if self._mask is None and use_set is not None:
            self._mask = Mask(use_set)
        cond1 = c is None
        cond2 = self._last_weight != weight
        cond3 = self._last_rho != rho
        cond4 = False
        if use_set is not None:
            if not np.alltrue(use_set == self._mask.use_set):
                cond4 = True
        if np.any([cond1, cond2, cond3, cond4]):
            n = len(v)
            I = np.eye(n)
            p = self._approx_period
            M = I[p:] - I[:-p]
            r = 2 * weight / rho
            ab = np.zeros((2, n))
            if use_set is None:
                I = np.eye(n)
            else:
                I = self._mask.MstM
            A = I + r * M.T.dot(M)
            for i in range(2):
                ab[i] = np.pad(np.diag(A, k=i), (0, i))
            c = spl.cholesky_banded(ab, lower=True)
            self._c = c
        return spl.cho_solve_banded((c, True), self._mask.zero_fill(v))