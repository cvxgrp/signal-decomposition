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

    def __init__(self):
        return

    @property
    def is_convex(self):
        return True

    @property
    def cost(self):
        diff2 = partial(cvx.diff, k=2)
        multiplier = lambda x: 1e3 * x
        cost = compose(multiplier, cvx.sum_squares, diff2)
        return cost

    @property
    def constraints(self):
        return []
