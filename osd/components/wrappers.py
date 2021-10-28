''' Component wrapper

This module contains wrappers to turn scalar component classes into vector
component classes


Author: Bennet Meyers
'''

import numpy as np

def make_columns_equal(component):
    class NewClass(component):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def _get_cost(self):
            f = super()._get_cost()
            def g(x):
                p = x.shape[1]
                return p * f(x)
            return g

        def prox_op(self, v, weight, rho):
            v_bar = np.average(v, axis=1)
            x_scalar = super().prox_op(v_bar, weight, rho * v.shape[1])
            x = np.tile(x_scalar, (v.shape[1], True)).T
            return x
    return NewClass

def make_columns_independent(component):
    class NewClass(component):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def _get_cost(self):
            f = super()._get_cost()
            def g(x):
                p = x.shape[1]
                return np.sum([f(x[:, i]) for i in range(p)])
            return g

        def prox_op(self, v, weight, rho):
            f = super().prox_op
            x_cols = [f(v[:, j], weight, rho)
                      for j in range(v.shape[1])]
            x = np.r_[x_cols].T
            return x
    return NewClass
