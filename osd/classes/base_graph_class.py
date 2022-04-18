''' Abstract base graph-form class module

This module contains the abstract base class for graph-form component classes

separable function key:

0: h(x) = 0
1: h(x) = |x|
2: h(x) = huber(x)
3: h(x) = { 0 if x = 0
          { 1 otherwise
4: h(x) = I(x >= 0)
5: h(x) = I(x <= 0)
6: h(x) = I(0 <= x <= 1)

Author: Bennet Meyers
'''

from abc import ABC, abstractmethod
import numpy as np
import scipy.sparse as sp
import itertools as itt

class GraphComponent(ABC):
    def __init__(self, weight, T, p, **kwargs):
        self._weight = weight
        self._T = T
        self._p = p
        self._x_size = T * p
        self.__set_z_size()
        for key in ['vmin', 'vmax', 'vavg', 'period', 'first_val']:
            if key in kwargs.keys():
                setattr(self, '_' + key, kwargs[key])
                del kwargs[key]
            else:
                setattr(self, '_' + key, None)
        self.__make_P()
        self._P = sp.block_diag([self._Px, self._Pz])
        self.__make_gz()
        self._gx = [{'f': 0, 'args': None, 'range': (0, self.x_size - 1)}]
        self._g = itt.chain.from_iterable([self._gx, self._gz])
        self.__make_A()
        self.__make_B()
        self.__make_c()
        # TODO: make all constraints their own classes and support chaining
        # self.__add_constraints()
        return

    def make_dict(self):
        canonicalized = {
            'P': self._P,
            'q': self._q,
            'r': self._r,
            'A': self._A,
            'B': self._B,
            'c': self._c,
            'g': self._g
        }
        return canonicalized

    def __set_z_size(self):
        self._z_size = 0

    def __make_P(self):
        self._Px = sp.dok_matrix(2 * (self.x_size))
        self._Pz = sp.dok_matrix(2 * (self.z_size))

    def __make_q(self):
        self._q = None

    def __make_r(self):
        self._r = None

    def __make_gz(self):
        self._gz = [{'f': 0,
                     'args': None,
                     'range': (self.x_size, self.x_size + self.z_size)}]

    def __make_A(self):
        self._A = sp.dok_matrix((0, self.x_size))

    def __make_B(self):
        self._B = sp.dok_matrix((0, self.z_size))

    def __make_c(self):
        self._c = np.zeros(0)

    def __add_constraints(self):
        if self._vmin is not None:
            # introduces new internal variable z
            self._z_size += self.x_size
            self._P = sp.block_diag([self._P,
                                        sp.dok_matrix(2 * (self.x_size,))])
            self._g = np.concatenate([self._g, 2 * np.ones(self.x_size)])
            self._A = sp.bmat(
                [[self._A],
                 [sp.eye(self.x_size)]]
            )
            self._B = sp.block_diag([self._B, -sp.eye(self.x_size)])
            self._c = np.concatenate([self._c,
                                      self._vmin * np.ones(self.x_size)])
        if self._vmax is not None:
            # introduces new internal variable z
            self._z_size += self.x_size
            self._P = sp.block_diag([self._P,
                                        sp.dok_matrix(2 * (self.x_size,))])
            self._g = np.concatenate([self._g, 2 * np.ones(self.x_size)])
            self._A = sp.bmat(
                [[self._A],
                 [sp.eye(self.x_size)]]
            )
            self._B = sp.block_diag([self._B, sp.eye(self.x_size)])
            self._c = np.concatenate([self._c,
                                      self._vmax * np.ones(self.x_size)])
        if self._vavg is not None:
            # introduces new constraints on x, but no new helper var
            newline = sp.coo_matrix(
                (np.ones(self.x_size),
                 (self.x_size * [1], np.arange(self.x_size)))
            )
            self._A = sp.bmat(
                [[self._A],
                 [newline]]
            )
            self._b = sp.bmat(
                [[self._A],
                 [sp.dok_matrix((1, self.z_size))]]
            )
            self._c = np.concatenate([self._c, [self._vavg]])

        if self._period is not None:
            # TODO: implement this
            pass
        if self._first_val is not None:
            # TODO: implement this
            pass



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



    def __add_constraints(self):


