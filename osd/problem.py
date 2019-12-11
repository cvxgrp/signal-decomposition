# -*- coding: utf-8 -*-
''' Data Handler Module

This module contains a class for defining a signal demixing optimization problem


Author: Bennet Meyers
'''

import numpy as np
import cvxpy as cvx
from itertools import chain

class Problem():
    def __init__(self, data, components):
        self.data = data
        self.components = components
        self.num_components = len(components)
        self.estimates = None

    def demix(self, solver='ECOS'):
        if np.alltrue([c().is_convex for c in self.components]):
            problem = self.__construct_cvx_problem()
            problem.solve(solver=solver)
            ests = [x.value for x in problem.variables()]
            self.estimates = ests
        else:
            raise NotImplemented

    def optimize_parameters(self):
        pass

    def holdout_validation(self):
        pass

    def __construct_cvx_problem(self):
        y = self.data
        T = len(y)
        K = self.num_components
        xs = [cvx.Variable(T) for _ in range(K)]
        costs = [c().cost(x) for c, x in zip(self.components, xs)]
        constraints = [c().constraints for c in self.components]
        constraints = list(chain.from_iterable(constraints))
        constraints.append(cvx.sum(xs) == y)
        objective = cvx.Minimize(cvx.sum(costs))
        problem = cvx.Problem(objective, constraints)
        return problem
