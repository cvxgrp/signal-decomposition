''' Frozen Component Module

This modules contains the class definition for a frozen component, or one in
which the estimate has been set ahead of time. This is given by the cost
function

ϕ(x) = I(x == x_frozen)

It has the very simple proximal operator of always returning x_frozen, regardless
of the input.

'''

import numpy as np
from osd.classes.component import Component

class Frozen(Component):

    def __init__(self, x_frozen, **kwargs):
        super().__init__(**kwargs)
        self.x_frozen = x_frozen
        self.x_frozen_periodic = None
        if 'period' in kwargs.keys():
            self.is_periodic = True
            self._internal_constraints = [
                lambda x, T, p: x[self.period:] == x[:-self.period],
                lambda x, T, p: x[:self.period] == self.x_frozen[:self.period]
            ]
        else:
            self.is_periodic = False
        return

    @property
    def is_convex(self):
        return True

    def _get_cost(self):
        cost = lambda x: 0
        return cost

    def prox_op(self, v, weight, rho, use_set=None, prox_weights=None):
        if not self.is_periodic:
            return self.x_frozen
        elif self.x_frozen_periodic is None:
            if len(v.shape) == 1:
                T = v.shape[0]
                p = 1
            else:
                T, p = v.shape
            length = self.period
            num_chunks = T // length
            if T % length != 0:
                num_chunks += 1
            if p == 1:
                out = np.tile(self.x_frozen[:self.period], num_chunks)
            else:
                out = np.tile(self.x_frozen[:self.period], (num_chunks, 1))
            out = out[:T]
            self.x_frozen_periodic = out
            return out
        else:
            return self.x_frozen_periodic