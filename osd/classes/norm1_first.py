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

import numpy as np
import cvxpy as cvx
from functools import partial
from osd.classes.component import Component
from osd.utilities import compose

class SparseFirstDiffConvex(Component):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._prox_prob = None
        return

    @property
    def is_convex(self):
        return True

    def _get_cost(self):
        diff1 = partial(cvx.diff, k=1)
        cost = compose(cvx.sum, cvx.abs, diff1)
        return cost

    def prox_op(self, v, weight, rho, use_set=None, prox_counts=None):
        # TODO: convert this to OSQP with custom canonicalization
        vec_in, weight_val, rho_val = np.copy(v), weight, rho
        vec_in[np.isnan(vec_in)] = 0
        problem = self._prox_prob
        if problem is None:
            n = len(vec_in)
            weight_over_rho = cvx.Parameter(value=weight_val / rho_val,
                                           name='weight_over_rho', pos=True)
            v = cvx.Parameter(n, value=vec_in, name='vec_in')
            x = cvx.Variable(n)
            if use_set is None:
                cost = (weight_over_rho * 2 * cvx.norm1(cvx.diff(x))
                        + cvx.sum_squares(x - v))
            else:
                cost = (weight_over_rho * 2 * cvx.norm1(cvx.diff(x))
                        + cvx.sum_squares(x[use_set] - v[use_set]))
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
        problem.solve(solver='MOSEK')
        return problem.variables()[0].value
