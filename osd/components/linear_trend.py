# -*- coding: utf-8 -*-
''' Linear Trend Component

This module contains the class for a linear trend with respect to time

Author: Bennet Meyers
'''

import cvxpy as cvx
import numpy as np
from osd.components.component import Component


class LinearTrend(Component):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.z = cvx.Variable(2)
        self._internal_constraints = [
            lambda x, T, K: x == np.c_[np.ones(T), np.arange(T)] @ self.z
        ]
        return

    @property
    def is_convex(self):
        return True

    def _get_cost(self):
        cost = lambda x: 0
        return cost
