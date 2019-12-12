# -*- coding: utf-8 -*-
''' Gaussian Noise Component

This module contains the class for Gaussian Noise

Author: Bennet Meyers
'''

import cvxpy as cvx
from osd.components.component import Component
from osd.utilities import compose

class LaplaceNoise(Component):

    def __init__(self):
        super().__init__()
        return

    @property
    def is_convex(self):
        return True

    def _get_cost(self):
        cost = compose(cvx.sum, cvx.abs)
        return cost
