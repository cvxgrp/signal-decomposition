import scipy.sparse as sp
import numpy as np
from gfosd.components.base_graph_class import GraphComponent

class FirstValEqual(GraphComponent):
    def __init__(self, value=0, *args, **kwargs):
        self._first_val = value
        super().__init__(*args, **kwargs)
        # always retain helper variable
        self._has_helpers = True

    def _make_A(self):
        super()._make_A()
        super()._make_B()
        super()._make_c()
        self._A = sp.bmat([
            [self._A.tocsr()[0]],
            [sp.dok_matrix((1, self._A.shape[1]))]
        ])

    def _make_B(self):
        self._B = sp.bmat([
            [self._B.tocsr()[0]],
            [sp.coo_matrix(([1], ([0], [0])), shape=(1, self._B.shape[1]))]
        ])

    def _make_c(self):
        self._c = np.concatenate([np.atleast_1d(self._c[0]),
                                  [self._first_val]])
        
class LastValEqual(GraphComponent):
    def __init__(self, value=0, *args, **kwargs):
        self._last_val = value
        super().__init__(*args, **kwargs)
        # always retain helper variable
        self._has_helpers = True

    def _make_A(self):
        super()._make_A()
        super()._make_B()
        super()._make_c()
        self._A = sp.bmat([
           [self._A.tocsr()[-1]],
            [sp.dok_matrix((1, self._A.shape[1]))]
        ])

    def _make_B(self):
        self._B = sp.bmat([
           [self._B.tocsr()[-1]],
            [sp.coo_matrix(([1], ([0], [self._B.shape[1]-1])), shape=(1, self._B.shape[1]))]
        ])

    def _make_c(self):
        self._c = np.concatenate([np.atleast_1d(self._c[-1]),
                                  [self._last_val]])
        
class AverageEqual(GraphComponent):
    def __init__(self, value=0, period=None, *args, **kwargs):
        self._avg_val = value
        self._period = period
        super().__init__(*args, **kwargs)
        # always retain helper variable, i.e. don't allow the Aggregate class
        # to try to remove variables.
        self._has_helpers = True

    def _set_z_size(self):
        if self._diff == 0:
            self._z_size = 0
        else:
            self._z_size = (self._T - self._diff) * self._p

    def _make_A(self):
        if self._diff == 0:
            if self._period is None:
                sum_len = self.x_size
            else:
                sum_len = self._period
            self._A = sp.csr_matrix(np.ones(sum_len), shape=(1, self.x_size))
        else:
            super()._make_A()
            super()._make_B()
            super()._make_c()
            self._A = sp.bmat([
                [self._A],
                [sp.dok_matrix((1, self._A.shape[1]))]
            ])

    def _make_B(self):
        if self._diff == 0:
            self._B = sp.dok_matrix((1, self.z_size))
        else:
            if self._period is None:
                sum_len = self.z_size
            else:
                sum_len = self._period
            self._B = sp.bmat([
                [self._B],
                [sp.csr_matrix(np.ones(sum_len), shape=(1, self.z_size))]
            ])

    def _make_c(self):
        if self._diff == 0:
            if self._period is None:
                sum_len = self.x_size
            else:
                sum_len = self._period
            self._c = np.atleast_1d(sum_len * self._avg_val)
        else:
            if self._period is None:
                sum_len = self.z_size
            else:
                sum_len = self._period
            self._c = np.concatenate([np.atleast_1d(self._c),
                                      [sum_len * self._avg_val]])

class NoCurvature(GraphComponent):
    def __init__(self, *args, **kwargs):
        super().__init__(diff=1, *args, **kwargs)

    def _make_P(self, size):
        return sp.dok_matrix(2 * (size + 1,))

    def _make_A(self):
        super()._make_A()
        super()._make_B()
        super()._make_c()
        self._A = sp.bmat([
            [self._A],
            [sp.dok_matrix((self.z_size, self._A.shape[1]))]
        ])

    def _make_B(self):
        self._B = sp.bmat([
            [self._B, sp.dok_matrix((self._B.shape[0], 1))],
            [sp.eye(self.z_size), sp.coo_matrix(
                -1 * np.ones(self.z_size).reshape((-1,1))
            )]
        ])

    def _make_c(self):
        self._c = np.concatenate([self._c, np.zeros(self.z_size)])
        self._z_size += 1

class NoSlope(GraphComponent):
    def __init__(self, *args, **kwargs):
        super().__init__(diff=0, *args, **kwargs)
        self._has_helpers = True

    def _set_z_size(self):
        self._z_size = 0

    def _make_A(self):
        T = self._T
        m1 = sp.eye(m=T - 1, n=T, k=0)
        m2 = sp.coo_matrix((-1 * np.ones(T-1),
                            (np.arange(T-1), T-1 * np.ones(T-1))),
                           shape=(T-1, T))
        self._A = m1 + m2
