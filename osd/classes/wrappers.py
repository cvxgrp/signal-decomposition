''' Component wrapper

This module contains wrappers to turn scalar component classes into vector
component classes


Author: Bennet Meyers
'''

import numpy as np
import cvxpy as cvx
import warnings

def make_columns_equal(component):
    class NewClass(component):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            if self.internal_constraints is None:
                self._internal_constraints = []
            if isinstance(self.internal_constraints, list):
                self._internal_constraints.append(
                    lambda x, T, p: cvx.diff(x, k=1, axis=1) == 0
                )
            else:
                # it's a function that generates a list of constraints
                ic = self._internal_constraints
                def new_f(x, T, p):
                    l1 = ic(x, T, p)
                    l1.append(cvx.diff(x, k=1, axis=1) == 0)
                    return l1
                self._internal_constraints = new_f

        def _get_cost(self):
            f = super()._get_cost()
            def g(x):
                return x.shape[1] * f(x[:, 0])
            return g

        def prox_op(self, v, weight, rho, use_set=None):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                v_bar = np.nanmean(v, axis=1)
            prox_counts = np.sum(~np.isnan(v), axis=1)
            prox_weights = prox_counts / v.shape[1]

            if use_set is not None:
                use_reduced = np.any(use_set, axis=1)
            else:
                use_reduced = ~np.isnan(v_bar)
            x_scalar = super().prox_op(v_bar, weight, rho,
                                       use_set=use_reduced,
                                       prox_weights=prox_weights)
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

        def prox_op(self, v, weight, rho, use_set=None):
            f = super().prox_op
            if use_set is not None:
                x_cols = [f(v[:, j], weight, rho, use_set=use_set[:, j])
                          for j in range(v.shape[1])]
            else:
                x_cols = [f(v[:, j], weight, rho, use_set=None)
                      for j in range(v.shape[1])]
            x = np.r_[x_cols].T
            return x
    return NewClass
