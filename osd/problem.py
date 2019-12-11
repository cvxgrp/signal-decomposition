# -*- coding: utf-8 -*-
''' Data Handler Module

This module contains a class for defining a signal demixing optimization problem


Author: Bennet Meyers
'''

import numpy as np
import cvxpy as cvx
from itertools import chain
import abc
from scipy.optimize import minimize_scalar

class Problem():
    def __init__(self, data, components, residual_term=0):
        self.data = data
        self.components = [c() if type(c) is abc.ABCMeta else c
                           for c in components]
        self.num_components = len(components)
        self.parameters = {i: c.parameters for i, c in enumerate(self.components)}
        self.num_parameters = np.sum([len(value) if value is not None else 0
                                      for key, value in self.parameters.items()])
        self.estimates = None
        self.problem = None
        self.residual_term = residual_term

    def demix(self, solver='ECOS', use_set=None, reset=False):
        if np.alltrue([c.is_convex for c in self.components]):
            if self.problem is None or reset:
                problem = self.__construct_cvx_problem(use_set=use_set)
                self.problem = problem
            else:
                problem = self.problem
            problem.solve(solver=solver)
            ests = [x.value for x in problem.variables()]
            self.estimates = ests
        else:
            raise NotImplemented

    def optimize_parameters(self, search_dict=None, solver='ECOS', seed=None):
        if seed is None:
            seed = np.random.random_integers(0, 1000)
        if self.num_parameters == 1:
            for key, value in self.parameters.items():
                if value is None:
                    continue
                else:
                    comp_ix = key
                    param = value[0]
            _ = self.holdout_validation(solver=solver, seed=seed)
            def cost_meta(v):
                val = 10 ** v
                self.components[comp_ix].set_parameters(val)
                cost = self.holdout_validation(solver=solver, seed=seed, reuse=True)
                return cost
            res = minimize_scalar(cost_meta)
            best_val = 10 ** res.x
            self.components[comp_ix].set_parameters(best_val)
            self.demix(solver=solver, reset=True)
            return
        else:
            print('IN PROGRESS')


    def holdout_validation(self, holdout=0.2, seed=None, solver='ECOS',
                               reuse=False):
        T = len(self.data)
        if seed is not None:
            np.random.seed(seed)
        hold_set = np.random.uniform(0, 1, T) <= holdout
        use_set = ~hold_set
        if not reuse:
            self.demix(solver=solver, use_set=use_set, reset=True)
        else:
            self.demix(solver=solver, reset=False)
        est_array = np.array(self.estimates)
        hold_est = np.sum(est_array[:, hold_set], axis=0)
        hold_y = self.data[hold_set]
        residuals = hold_y - hold_est
        resid_cost =self.components[self.residual_term].cost
        holdout_cost = resid_cost(residuals).value
        return holdout_cost.item()

    def __construct_cvx_problem(self, use_set=None):
        if use_set is None:
            use_set = np.s_[:]
        y = self.data
        T = len(y)
        K = self.num_components
        xs = [cvx.Variable(T) for _ in range(K)]
        costs = [c.cost(x) for c, x in zip(self.components, xs)]
        constraints = [c.constraints for c in self.components]
        constraints = list(chain.from_iterable(constraints))
        constraints.append(cvx.sum([x[use_set] for x in xs], axis=0) == y[use_set])
        objective = cvx.Minimize(cvx.sum(costs))
        problem = cvx.Problem(objective, constraints)
        return problem
