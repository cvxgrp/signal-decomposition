# -*- coding: utf-8 -*-
''' Gaussian Noise Component

This module contains the class for Gaussian Noise

Author: Bennet Meyers
'''

import cvxpy as cvx
from osd.components.component import Component

class GaussNoise(Component):

    def __init__(self):
        return

    @property
    def is_convex(self):
        return True

    @property
    def cost(self):
        return cvx.sum_squares

    @property
    def constraints(self):
        return None