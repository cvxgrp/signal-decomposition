# -*- coding: utf-8 -*-
''' Gaussian Noise Component

This module contains the class for Gaussian Noise

Author: Bennet Meyers
'''

import cvxpy as cvx
from osd.components.component import Component
from osd.utilities import compose

class LaplaceNoise(Component):

    def __init__(self, gamma=1):
        super().__init__(gamma)
        return

    @property
    def is_convex(self):
        return True

    def _get_cost(self):
        gamma = self.parameters[0]
        multiplier = lambda x: gamma * x
        cost = compose(multiplier, cvx.sum, cvx.abs)
        return cost

    def _get_params(self):
        gamma = cvx.Parameter(nonneg=True)
        return [gamma]