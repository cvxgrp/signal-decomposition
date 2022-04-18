''' Gaussian Noise Component

This module contains the class for Gaussian Noise

Author: Bennet Meyers
'''

import cvxpy as cvx
import numpy as np
import scipy.sparse as sp
from osd.classes.component import Component
from osd.classes.base_graph_class import GraphComponent

class MeanSquareSmall(Component):

    def __init__(self, size=1, **kwargs):
        super().__init__(**kwargs)
        self.size = size
        return

    @property
    def is_convex(self):
        return True

    def _get_cost(self):
        f = lambda x: cvx.sum_squares(x) / self.size
        return f

    def prox_op(self, v, weight, rho, use_set=None, prox_weights=None):
        a = (2 * weight) / (rho * self.size)
        if prox_weights is not None:
            a /= prox_weights
        r = 1 / (1 + a)
        out = r * np.asarray(v)
        if use_set is not None:
            out[~use_set] = 0
        return out

    def make_graph_form(self, T, p):
        gf = MeanSquareSmallGraph(
            self.weight, T, p,
            vmin=self.vmin, vmax=self.vmax,
            period=self.period, first_val=self.first_val
        )
        self._gf = gf
        return gf.make_dict()

class MeanSquareSmallGraph(GraphComponent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        return

    def __make_P(self):
        self._Px = (self.weight/self.x_size) * sp.eye(self.x_size)
        self._Pz = sp.dok_matrix(2 * (self.z_size))