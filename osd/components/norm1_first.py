# -*- coding: utf-8 -*-
''' Smooth Component (1)

This module contains the class for the convex heuristic for a piecewise constant
function. A piecewise constant function has a sparse first-order difference;
many differences are exactly zero and a small number of them can be large.

A convex approximation of this problem is minimizing the L1-norm of the first-
order difference:

    minimize || D x ||_1

This is also known as a Total Variation filter
    (https://en.wikipedia.org/wiki/Total_variation_denoising)

Author: Bennet Meyers
'''

import cvxpy as cvx
from functools import partial
from osd.components.component import Component
from osd.utilities import compose

class SparseFirstDiffConvex(Component):

    def __init__(self):
        super().__init__()
        return

    @property
    def is_convex(self):
        return True

    def _get_cost(self):
        diff1 = partial(cvx.diff, k=1)
        cost = compose(cvx.sum, cvx.abs, diff1)
        return cost
