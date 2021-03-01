# -*- coding: utf-8 -*-
''' Smooth Component (2)

This module contains the class for a smooth signal that is penalized for large
second-order differences

Author: Bennet Meyers
'''

import cvxpy as cvx
from functools import partial
from osd.components.component import Component
from osd.utilities import compose

class SmoothSecondDifference(Component):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    @property
    def is_convex(self):
        return True

    def _get_cost(self):
        internal_scaling = 1 #10 ** (3.5 / 2)
        diff2 = partial(cvx.diff, k=2)
        cost = compose(lambda x: internal_scaling * x, diff2)
        cost = compose(cvx.sum_squares, cost)
        return cost
