# -*- coding: utf-8 -*-
''' Gaussian Noise Component

This module contains the class for a smooth signal that is penalized for large
second-order differences

Author: Bennet Meyers
'''

import cvxpy as cvx
from functools import partial
from osd.components.component import Component
from osd.utilities import compose

class SmoothSecondDifference(Component):

    def __init__(self, gamma=1e3):
        super().__init__(gamma)
        return

    @property
    def is_convex(self):
        return True

    def _get_cost(self):
        diff2 = partial(cvx.diff, k=2)
        gamma = self.parameters[0]
        multiplier = lambda x: gamma * x
        cost = compose(multiplier, cvx.sum_squares, diff2)
        return cost

    def _get_params(self):
        gamma = cvx.Parameter(nonneg=True)
        return [gamma]


