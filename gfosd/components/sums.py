import numpy as np
import scipy.sparse as sp
from gfosd.components.base_graph_class import GraphComponent


class SumSquare(GraphComponent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        return

    def _make_P(self, size):
        return self.weight * 2 * sp.eye(size)  # note the (1/2) in canonical form!


class SumSquareReference(GraphComponent):
    """
    phi(x;v) = || x - v ||_2^2 = ||x||_2^2 -2v^Tx + ||v||_2^2
    """

    def __init__(self, vec, mask=None, *args, **kwargs):
        self._vec = vec
        self._mask = mask
        super().__init__(*args, **kwargs)
        if self._mask is not None:
            self._has_helpers = True

    def _set_z_size(self):
        if self._mask is None:
            self._z_size = (self._T - self._diff) * self._p
        else:
            self._z_size = self._mask.shape[0]

    def _make_P(self, size):
        return self.weight * 2 * sp.eye(size)  # note the (1/2) in canonical form!

    def _make_q(self):
        if self._mask is None:
            qx = self.weight * (-2) * self._vec
            self._q = np.r_[qx, np.zeros(self.z_size)]
        else:
            qz = (-2) * self.weight * self._mask @ self._vec
            self._q = np.r_[np.zeros(self.x_size), qz]

    def _make_r(self):
        if self._mask is None:
            self._r = self.weight * np.sum(np.square(self._vec))
        else:
            self._r = self.weight * np.sum(np.square(self._mask @ self._vec))

    def _make_A(self):
        self._A = self._mask


class SumAbs(GraphComponent):
    def __init__(self, weighting_mat=None, *args, **kwargs):
        self.weighting_mat = weighting_mat
        super().__init__(*args, **kwargs)
        return

    def _make_B(self):
        if self.weighting_mat is None:
            self._B = sp.eye(self._A.shape[0], self.z_size) * -1
        else:
            self._B = self.weighting_mat

    def _make_g(self, size):
        g = [{'g': 'abs',
              'args': {'weight': self.weight},
              'range': (0, size)}]
        return g


class SumHuber(GraphComponent):
    def __init__(self, M=1, *args, **kwargs):
        self._M = M
        super().__init__(*args, **kwargs)
        return

    def _make_g(self, size):
        g = [{'g': 'huber',
              'args': {'weight': self.weight, 'M': self._M},
              'range': (0, size)}]
        return g


class SumQuantile(GraphComponent):
    def __init__(self, tau, *args, **kwargs):
        self.tau = tau
        super().__init__(*args, **kwargs)
        return

    def _make_g(self, size):
        g = [{'g': 'quantile',
              'args': {'weight': self.weight, 'tau': self.tau},
              'range': (0, size)}]
        return g


class SumCard(GraphComponent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        return

    def _make_g(self, size):
        g = [{'g': 'card',
              'args': {'weight': self.weight},
              'range': (0, size)}]
        return g
