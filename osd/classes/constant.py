''' Constant Component

This module contains the class for a constant signal

Author: Bennet Meyers
'''

import numpy as np
import cvxpy as cvx
from functools import partial
from osd.classes.component import Component
import warnings

class Constant(Component):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._internal_constraints = [
            lambda x, T, p: cvx.diff(x, k=1) == 0
        ]
        return

    @property
    def is_convex(self):
        return True

    def _get_cost(self):
        cost = lambda x: 0
        return cost

    def prox_op(self, v, weight, rho, use_set=None, prox_counts=None):
        if use_set is None:
            avg = np.average(v)
        else:
            avg = np.average(v[use_set])
        return np.ones_like(v) * avg

class ConstantChunks(Component):
    def __init__(self, length, use_set=None, **kwargs):
        super().__init__(**kwargs)
        self.length=length
        if use_set is not None and len(use_set.shape) > 1:
            self.use_set = np.alltrue(use_set, axis=1)
        else:
            self.use_set = use_set
        self._internal_constraints = partial(make_constraints,
                                             length=self.length)
        return

    @property
    def is_convex(self):
        return True

    def _get_cost(self):
        cost = lambda x: 0
        return cost

    def prox_op(self, v, weight, rho, use_set=None, prox_weights=None):
        if use_set is not None:
            self.use_set = use_set
        T = len(v)
        temp = v.copy()
        if self.use_set is not None:
            temp[~self.use_set] = np.nan
        num_chunks = T // self.length
        if T % self.length != 0:
            num_chunks += 1
            pad = self.length - T % self.length
            temp = np.r_[temp, [np.nan] * pad]
        if prox_weights is not None:
            pc  = prox_weights.copy()
            if len(pc) < len(temp):
                pc = np.r_[pc, [0] * pad]
                temp *= np.r_[prox_weights, [0] * pad]
            else:
                temp *= prox_weights
            pc = pc.reshape((self.length, num_chunks), order='F')
            col_counts = np.sum(pc, axis=0)
        temp = temp.reshape((self.length, num_chunks), order='F')
        # print(temp.shape)
        nrows = temp.shape[0]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            avgs = np.nanmean(temp, axis=0)
        avgs[np.isnan(avgs)] = 0
        if prox_weights is not None:
            # note the special handling of nan values at this point!
            avgs *= np.sum(~np.isnan(temp), axis=0)
            avgs /= col_counts
        out = np.tile(avgs, (nrows, True))
        out = out.ravel(order='F')
        out = out[:len(v)]
        return out

def make_constraints(x, T, K, length=None):
    num_chunks = T // length
    if T % length != 0:
        num_chunks += 1
    constraints = []
    for i in range(num_chunks):
        constraints.append(cvx.diff(x[i*length:(i+1)*length], k=1) == 0)
    return constraints