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
import numpy as np
from osd.components.component import Component
from osd.utilities import compose

class SparseSecondDiffConvex(Component):

    def __init__(self, internal_scale=1., **kwargs):
        super().__init__(**kwargs)
        self._prox_prob = None
        self.internal_scale = internal_scale
        return

    @property
    def is_convex(self):
        return True

    def _get_cost(self):
        diff2 = partial(cvx.diff, k=2)
        cost = compose(cvx.sum, cvx.abs, lambda x: self.internal_scale * x, diff2)
        return cost

    def prox_op(self, vec_in, weight_val, rho_val, solver='MOSEK',
                verbose=False):
        problem = self._prox_prob
        if problem is None:
            n = len(vec_in)
            weight_over_rho = cvx.Parameter(value=weight_val / rho_val,
                                           name='weight_over_rho', pos=True)
            v = cvx.Parameter(n, value=vec_in, name='vec_in')
            x = cvx.Variable(n)
            cost = weight_over_rho * 2 * cvx.norm1( self.internal_scale *
                cvx.diff(x, k=2)) + cvx.sum_squares(x - v)
            c = []
            if self.vmin is not None:
                c.append(x >= self.vmin)
            if self.vmax is not None:
                c.append(x <= self.vmax)
            if self.vavg is not None:
                n = x.size
                c.append(cvx.sum(x) / n == self.vavg)
            if self.period is not None:
                p = self.period
                c.append(x[:-p] == x[p:])
            if self.first_val is not None:
                c.append(x[0] == self.first_val)
            problem = cvx.Problem(cvx.Minimize(cost), c)
            self._prox_prob = problem
        else:
            parameters = {p.name(): p for p in problem.parameters()}
            parameters['vec_in'].value = vec_in
            if ~np.isclose(weight_val / rho_val,
                           parameters['weight_over_rho'].value,
                           atol=1e-3):
                parameters['weight_over_rho'].value = weight_val / rho_val
        problem.solve(solver=solver, verbose=verbose)
        return problem.variables()[0].value
