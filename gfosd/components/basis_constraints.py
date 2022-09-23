"""
Basis Class

This encodes the constraint that the component is representable in basis form,
i.e

    x = Az,

where A is some basis matrix. This also includes any additional penalty on z
that is avaible in the menu of g functions.

"""


import numpy as np
import scipy.sparse as sp
from gfosd.components.base_graph_class import GraphComponent

class Basis(GraphComponent):
    def __init__(self, basis, penalty=None, *args, **kwargs):
        self._basis = basis
        self._penalty = penalty
        super().__init__(*args, **kwargs)
        self._has_helpers = True

    def _set_z_size(self):
        self._z_size = self._basis.shape[1]

    def _make_B(self):
        # if self._basis.shape[0] !=
        self._B = self._basis * -1

    def _make_g(self, size):
        if (self._penalty is None) or (self._penalty == 'sum_square'):
            g = []
        else:
            g = [{'g': self._penalty,
                         'args': {'weight': self.weight},
                         'range': (0, size)}]
        return g

    def _make_P(self, size):
        if (self._penalty is None) or (self._penalty != 'sum_square'):
            P = sp.dok_matrix(2 * (size,))
        else:
            P = self.weight * sp.eye(size)
        return P

class Periodic(Basis):
    def __init__(self, period, *args, **kwargs):
        self._period = period
        T = int(T)
        num_periods = int(np.ceil(T / period))
        M = sp.eye(period)
        basis = sp.vstack([M] * num_periods)
        basis = basis.tocsr()
        basis = basis[:T]
        super().__init__(basis, *args, **kwargs)
