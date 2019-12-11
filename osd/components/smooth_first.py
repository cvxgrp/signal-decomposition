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

    def __init__(self, gamma=1e1):
        super().__init__(gamma)
        return

    @property
    def is_convex(self):
        return True

    def _get_cost(self):
        diff2 = partial(cvx.diff, k=1)
        gamma = self.parameters[0]
        multiplier = lambda x: gamma * x
        cost = compose(multiplier, cvx.sum_squares, diff2)
        return cost

    def _get_params(self):
        gamma = cvx.Parameter(nonneg=True)
        return [gamma]
