''' Constant Component

This module contains the class for a constant signal

Author: Bennet Meyers
'''

import numpy as np
import cvxpy as cvx
from functools import partial
from osd.components.component import Component

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

    def prox_op(self, v, weight, rho):
        avg = np.average(v)
        return avg

class ConstantChunks(Component):
    def __init__(self, length, **kwargs):
        super().__init__(**kwargs)
        self.length=length
        self._internal_constraints = partial(make_constraints,
                                             length=self.length)
        return

    @property
    def is_convex(self):
        return True

    def _get_cost(self):
        cost = lambda x: 0
        return cost

    def prox_op(self, v, weight, rho):
        T = len(v)
        temp = v.copy()
        num_chunks = T // self.length
        if T % self.length != 0:
            num_chunks += 1
            pad = self.length - T % self.length
            temp = np.r_[temp, [np.nan] * pad]
        temp = temp.reshape((self.length, num_chunks), order='F')
        # print(temp.shape)
        nrows = temp.shape[0]
        avgs = np.nanmean(temp, axis=0)
        out = np.tile(avgs, (nrows, True))
        out = out.ravel(order='F')
        out = out[:len(v)]
        # return np.clip(out, -np.inf, 0)
        return out

def make_constraints(x, T, K, length=None):
    num_chunks = T // length
    if T % length != 0:
        num_chunks += 1
    constraints = []
    for i in range(num_chunks):
        constraints.append(cvx.diff(x[i*length:(i+1)*length], k=1) == 0)
    return constraints