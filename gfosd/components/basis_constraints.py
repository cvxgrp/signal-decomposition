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

    def _make_B(self):
        self._B = self._basis * -1

    def _make_gz(self):
        if (self._penalty is None) or (self._penalty == 'sum_square'):
            self._gz = []
        else:
            self._gz = [{'g': self._penalty,
                         'args': {'weight': self.weight},
                         'range': (0, self.z_size)}]

    def _make_P(self):
        if (self._penalty is None) or (self._penalty != 'sum_square'):
            self._Pz = sp.dok_matrix(2 * (self.z_size,))
        else:
            self._Pz = self.weight * sp.eye(self.z_size)

class Periodic(Basis):
    def __init__(self, period, T, *args, **kwargs):
        self._period = period
        num_periods = np.ceil(T / period)
        M = sp.eye(period)
        basis = sp.vstack([M] * num_periods)
        basis = basis.tocsr()
        basis = basis[:T]
        super().__init__(basis, T, *args, **kwargs)