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
        basis_len = self._basis.shape[0]
        # below allows for lazy evaluation of the full basis dictionary,
        # allowing for the user to define a basis based on a known shorter
        # period without knowing the signal length ahead of time
        if basis_len != self._T:
            num_periods = int(np.ceil(self._T / basis_len))
            M = sp.vstack([self._basis] * num_periods)
            M = M.tocsr()
            M = M[:self._T]
            self._basis = M
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
        M = sp.eye(period)
        super().__init__(M, *args, **kwargs)


class Fourier(Basis):
    def __init__(self, periods, num_harmonics, num_harmonics_cross=None, *args, **kwargs):
        self._periods = np.atleast_1d(periods)
        self._num_harmonics = num_harmonics
        self._num_harmonics_cross = num_harmonics_cross
        max_per = max(self._periods)
        M = make_fourier_basis_cross(max_per, self._periods, self._num_harmonics, self._num_harmonics_cross)
        super().__init__(M, *args, **kwargs)

    def _make_B(self):
        self._basis = make_fourier_basis_cross(self._T, self._periods, self._num_harmonics, self._num_harmonics_cross)
        self._B = self._basis * -1


def make_fourier_basis(T, p, K):
    """
    T: length of signal
    p: length of period
    K: number of harmonics
    """
    ts = np.arange(T)
    sines = [np.sin(2 * (kix + 1) * np.pi * ts / p) for kix in range(K)]
    cosines = [np.cos(2 * (kix + 1) * np.pi * ts / p) for kix in range(K)]
    B = np.r_[sines, cosines].T
    B /= np.max(B, axis=0)
    return B


def make_fourier_basis_cross(T, ps, K, crossK=None):
    """
    T: length of signal
    ps: list of period lengths
    K: number of harmonics
    crossK: number of harmonics to keep in cross terms
    """
    if crossK is None:
        crossK = min(2 * K, 2 * 8)
    ps = np.atleast_1d(ps)
    num_ps = len(ps)
    wfs = [make_fourier_basis(T, p, K) for p in ps]
    if num_ps == 1:
        return np.r_[np.ones(T)[np.newaxis, :] / np.sqrt(T), wfs[0].T].T
    elif num_ps == 2:
        wfs_cross = np.einsum('ij,ik->ijk', wfs[0][:, :crossK], wfs[1][:, :crossK])
        wfs_cross = wfs_cross.reshape((T, -1))
        wfs_cross /= np.max(wfs_cross, axis=0)
        return np.r_[np.ones(T)[np.newaxis, :] / np.sqrt(T), wfs[0].T, wfs[1].T, wfs_cross.T].T
    elif num_ps == 3:
        wfs_cross = np.einsum('ij,ik,il->ijkl', wfs[0][:, :crossK], wfs[1][:, :crossK], wfs[2][:, :crossK])
        wfs_cross = wfs_cross.reshape((T, -1))
        wfs_cross /= np.max(wfs_cross, axis=0)
        return np.r_[np.ones(T)[np.newaxis, :] / np.sqrt(T), wfs[0].T, wfs[1].T, wfs[2].T, wfs_cross.T].T
