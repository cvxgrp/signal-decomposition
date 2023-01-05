''' Abstract base graph-form class module

This module contains the abstract base class for graph-form component classes

separable function key:

0:          g(x) = 0
abs:        g(x) = |x|
huber:      g(x) = huber(x)
card:       g(x) = { 0 if x = 0
                   { 1 otherwise
is_pos:     g(x) = I(x >= 0)
is_neg:     g(x) = I(x <= 0)
is_bound:   g(x) = I(0 <= x <= 1)
is_finite_set: g(x) = I(x ∈ S)

Author: Bennet Meyers
'''

from abc import ABC, abstractmethod
import numpy as np
import scipy.sparse as sp
import itertools as itt

class GraphComponent(ABC):
    def __init__(self, weight=1, diff=0, **kwargs):
        self._weight = weight
        self._diff = diff
        self._has_helpers = diff != 0
        self._Px = None
        self._Pz = None
        self._q = None  # not currently used
        self._r = None  # not currently used
        self._A = None
        self._B = None
        self._c = None
        self._gx = None
        self._gz = None
        return

    def prepare_attributes(self, T, p=1):
        self._T = T
        self._p = p
        self._x_size = T * p
        # There are two ways to construct the problem data for a class for QSS.
        # The first way applies the quadratic and separable costs to a helper
        # variable. The second way applies the costs to the component variable.
        # Typically, we want to avoid using helper variables if they aren't
        # necessary (e.g. for x = z relationships). But in the Aggregate
        # component istances, we need to be careful, because we can't point
        # more than one g to a given component variable. THis is handles in the
        # Aggregate class.
        if self._has_helpers:
            self._set_z_size()
            self._Px = sp.dok_matrix(2 * (self.x_size,))
            self._Pz = self._make_P(self.z_size)
            self._make_q()
            self._make_r()
            self._gx = []
            self._gz = self._make_g(self.z_size)
            self._make_A()
            self._make_B()
            self._make_c()
        else:
            self._z_size = 0
            self._Px = self._make_P(self.x_size)
            self._Pz = sp.dok_matrix(2 * (self.z_size,))
            self._make_q()
            self._make_r()
            self._gx = self._make_g(self.x_size)
            self._gz = []
            self._A = sp.dok_matrix((0, self.x_size))
            self._B = sp.dok_matrix((0, self.z_size))
            self._c = np.zeros(0)

    def make_dict(self):
        canonicalized = {
            'Px': self._Px,
            'Pz': self._Pz,
            'q': self._q, #not currently used
            'r': self._r, #not currently used
            'A': self._A,
            'B': self._B,
            'c': self._c,
            'gx': self._gx,
            'gz': self._gz
        }
        return canonicalized

    def _set_z_size(self):
        self._z_size = (self._T - self._diff) * self._p

    def _make_P(self, size):
        return sp.dok_matrix(2 * (size,))

    def _make_q(self):
        self._q = None

    def _make_r(self):
        self._r = None

    def _make_g(self, size):
        return []

    def _make_A(self):
        if self._diff == 0:
            self._A = sp.eye(self.x_size)
        elif self._diff == 1:
            T = self._T
            m1 = sp.eye(m=T - 1, n=T, k=0)
            m2 = sp.eye(m=T - 1, n=T, k=1)
            self._A = m2 - m1
        elif self._diff == 2:
            T = self._T
            m1 = sp.eye(m=T - 2, n=T, k=0)
            m2 = sp.eye(m=T - 2, n=T, k=1)
            m3 = sp.eye(m=T - 2, n=T, k=2)
            self._A = m1 - 2 * m2 + m3
        elif self._diff == 3:
            T = self._T
            m1 = sp.eye(m=T - 3, n=T, k=0)
            m2 = sp.eye(m=T - 3, n=T, k=1)
            m3 = sp.eye(m=T - 3, n=T, k=2)
            m4 = sp.eye(m=T - 3, n=T, k=3)
            self._A = -m1 + 3 * m2 - 3 * m3 + m4
        else:
            print('Differences higher than 3 not supported')
            raise Exception


    def _make_B(self):
        self._B = sp.eye(self._A.shape[0], self.z_size) * -1

    def _make_c(self):
        self._c = np.zeros(self._B.shape[0])


    @property
    def weight(self):
        return self._weight

    def set_weight(self, weight):
        self._weight = weight
        return

    @property
    def T(self):
        return self._T

    @property
    def p(self):
        return self._p

    @property
    def size(self):
        return self.x_size + self.z_size

    @property
    def x_size(self):
        return self._x_size

    @property
    def z_size(self):
        return self._z_size

    @property
    def P_x(self):
        return self._P
