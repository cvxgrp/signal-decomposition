# -*- coding: utf-8 -*-
''' Linear Trend Component

This module contains the class for a linear trend with respect to time

Author: Bennet Meyers
'''

import cvxpy as cvx
from osd.components.component import Component


class LinearTrend(Component):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    @property
    def is_convex(self):
        return True

    def _get_cost(self):
        slope = cvx.Variable()
        cost = lambda x: None
        return
