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
import scipy.sparse as sp
import cvxpy as cvx
from functools import partial
import warnings
from osd.classes.component import Component
from osd.utilities import compose
from osd.classes.base_graph_class import GraphComponent

class SparseFirstDiffConvex(Component):

    def __init__(self, internal_scale=1., solver=None, **kwargs):
        super().__init__(**kwargs)
        self._prox_prob = None
        self.internal_scale = internal_scale
        self._solver = solver
        self._last_set = None
        return

    @property
    def is_convex(self):
        return True

    def _get_cost(self):
        diff1 = partial(cvx.diff, k=1)
        cost = compose(cvx.sum, cvx.abs, diff1)
        return cost

    def prox_op(self, v, weight, rho, use_set=None, verbose=False,
                prox_counts=None):
        if use_set is None:
            use_set = np.ones_like(v, dtype=bool)
        problem = self._prox_prob
        ic = self.internal_scale
        if self._last_set is not None:
            set_change = ~np.alltrue(use_set == self._last_set)
        else:
            set_change = True
        if problem is None or set_change:
            x = cvx.Variable(len(v))
            Mv = cvx.Parameter(np.sum(use_set), value=v[use_set], name='Mv')
            w = cvx.Parameter(value=weight, name='weight', nonneg=True)
            r = cvx.Parameter(value=rho, name='rho', nonneg=True)
            objective = cvx.Minimize(
                w * cvx.norm1(ic * cvx.diff(x, k=1)) + r / 2 * cvx.sum_squares(
                    x[use_set] - Mv
                )
            )
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
            problem = cvx.Problem(objective, c)
            self._prox_prob = problem
            self._last_set = use_set
        else:
            params = problem.param_dict
            params['Mv'].value = v[use_set]
            if ~np.isclose(weight, params['weight'].value, atol=1e-3):
                params['weight'].value = weight
            if ~np.isclose(rho, params['rho'].value, atol=1e-3):
                params['rho'].value = rho
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            problem.solve(solver=self._solver)
        return problem.variables()[0].value

    def make_graph_form(self, T, p):
        gf = SparseFirstDiffConvexGraph(
            self.weight, T, p,
            vmin=self.vmin, vmax=self.vmax,
            period=self.period, first_val=self.first_val
        )
        self._gf = gf
        return gf.make_dict()

class SparseFirstDiffConvexGraph(GraphComponent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        return

    def __set_z_size(self):
        self._z_size = self.x_size - 1

    def __make_gz(self):
        self._gz = [{'f': 1,
                     'args': None,
                     'range': (self.x_size, self.x_size + self.z_size)}]

    def __make_A(self):
        T = self._T
        m1 = sp.eye(m=T - 1, n=T, k=0)
        m2 = sp.eye(m=T - 1, n=T, k=1)
        D = m2 - m1
        self._A = D * (self.weight)