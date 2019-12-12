# -*- coding: utf-8 -*-
''' Smooth Component (1)

This module contains the class for the convex heuristic for a piecewise linear
function. A piecewise constant function has a sparse second-order difference;
many changes in slope are exactly zero and a small number of them can be large.

A convex approximation of this problem is minimizing the L1-norm of the second-
order difference:

    minimize || D_2 x ||_1

This is an extension of the concept of Total Variation filtering, applied to the
differences of a discrete signal, rather than the values.

Author: Bennet Meyers
'''

import cvxpy as cvx
from functools import partial
from osd.components.component import Component
from osd.utilities import compose

class SparseSecondDiffConvex(Component):

    def __init__(self):
        super().__init__()
        return

    @property
    def is_convex(self):
        return True

    def _get_cost(self):
        diff2 = partial(cvx.diff, k=2)
        cost = compose(cvx.sum, cvx.abs, diff2)
        return cost
