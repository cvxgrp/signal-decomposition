# -*- coding: utf-8 -*-
''' Smooth Component (1)

This module contains the class for a smooth signal that is penalized for large
first-order differences

Author: Bennet Meyers
'''

import cvxpy as cvx
from functools import partial
from osd.components.component import Component
from osd.utilities import compose

class SmoothFirstDifference(Component):

    def __init__(self):
        super().__init__()
        return

    @property
    def is_convex(self):
        return True

    def _get_cost(self):
        diff1 = partial(cvx.diff, k=1)
        cost = compose(cvx.sum_squares, diff1)
        return cost
