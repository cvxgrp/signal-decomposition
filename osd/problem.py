# -*- coding: utf-8 -*-
''' SD Problem Module

This module contains a class for defining a signal demixing optimization problem


Author: Bennet Meyers
'''

import numpy as np
import cvxpy as cvx
from itertools import chain
import abc
from scipy.optimize import minimize_scalar
from sklearn.model_selection import train_test_split
from osd.signal_decomp_admm import run_admm
from osd.signal_decomp_bcd import run_bcd
from osd.utilities import compose, calc_obj
import matplotlib.pyplot as plt

class Problem():
    def __init__(self, data, classes, residual_term=0):
        self.data = data
        if len(data.shape) == 1:
            self.T = data.shape[0]
            self.p =1
        else:
            self.T, self.p = data.shape
        self.classes = [c() if type(c) is abc.ABCMeta else c
                        for c in classes]
        self.K = len(classes)
        self.parameters = {i: c.parameters for i, c in enumerate(self.classes)
                           if c.parameters is not None}
        self.num_parameters = int(
            np.sum([len(value) if value is not None else 0
                    for key, value in self.parameters.items()])
        )
        self.components = None
        self.problem = None
        self.admm_result = None
        self.bcd_result = None
        self.residual_term = residual_term # typically 0
        self.known_set = ~np.isnan(data)
        self.q = np.sum(self.known_set)
        # CVXPY objects (not used for ADMM)
        self.__weights = cvx.Parameter(shape=self.K, nonneg=True,
                                       value=[c.weight for c in self.classes])
        self.use_set = None

    def __repr__(self):
        st = " SD problem instance at {} (K={}, T={}, p={}, q={})>"
        if self.is_convex:
            st = "<convex" + st
        else:
            st = "<nonconvex" + st
        return st.format(hex(id(self)), self.K, self.T, self.p, self.q)

    @property
    def objective_value(self):
        if self.components is not None:
            obj_val = calc_obj(self.data, self.components, self.classes,
                               self.use_set)
            return obj_val
        else:
            return

    @property
    def weights(self):
        return self.__weights.value

    @property
    def is_convex(self):
        return np.alltrue([c.is_convex for c in self.classes])

    def decompose(self, use_set=None, rho=None, rho0_scale=None, how=None,
                  num_iter=1e3, verbose=True, reset=False,
                  X_init=None, u_init=None,
                  stop_early=True, abs_tol=1e-5, rel_tol=1e-5,
                  **cvx_kwargs):
        if rho0_scale is None:
            rho0_scale = 0.7
        if rho is None:
            rho = 2 / (self.data.size * self.classes[0].weight)
        num_iter = int(num_iter)
        if use_set is None:
            use_set = self.known_set
        else:
            use_set = np.logical_and(use_set, self.known_set)
        self.use_set = use_set
        self.set_weights([c.weight for c in self.classes])
        if how is None:
            if self.is_convex:
                how = 'bcd'
                if verbose:
                    print('Convex problem detected. Using BCD...')
            else:
                how = 'admm-polish'
                if verbose:
                    print('Non-convex problem detected. Using ADMM with ' +
                          'BCD polish...')
        if np.all([
            X_init is None,
            reset is False,
            self.components is not None,
        ]):
            X_init = self.components

        if self.is_convex and how.lower() in ['cvx', 'cvxpy']:
            if self.problem is None or reset or np.any(use_set != self.use_set):
                problem = self.__construct_cvx_problem(use_set=use_set)
                self.problem = problem
            else:
                problem = self.problem
            if X_init is not None:
                cvx_kwargs['warm_start'] = True
                for ix, x in enumerate(problem.variables()):
                    x.value = X_init[ix, :]
            # print(self.problem.is_dcp())
            problem.solve(verbose=verbose, **cvx_kwargs)
            sorted_order = np.argsort([v.name() for v in problem.variables()])
            ests = np.array([x.value for x in
                             np.asarray(problem.variables())[sorted_order]
                             if 'x_' in x.name()])
            self.components = ests
        elif how.lower() in ['admm', 'sd-admm']:
            result = run_admm(
                self.data, self.classes, num_iter=num_iter, rho=rho*rho0_scale,
                use_ix=use_set, verbose=verbose, X_init=X_init, u_init=u_init,
                stop_early=stop_early, abs_tol=abs_tol, rel_tol=rel_tol,
                residual_term=self.residual_term
            )
            self.admm_result = result
            self.components = result['X']
        elif how.lower() in ['bcd', 'sd-bcd']:
            result = run_bcd(
                self.data, self.classes, num_iter=num_iter, use_ix=use_set,
                abs_tol=abs_tol, rel_tol=rel_tol, X_init=X_init,
                verbose=verbose
            )
            self.bcd_result = result
            self.components = result['X']
        elif how.lower() in ['admm-polish', 'admm-bcd']:
            result = run_admm(
                self.data, self.classes, num_iter=num_iter, rho=rho*rho0_scale,
                use_ix=use_set, verbose=verbose, X_init=X_init, u_init=u_init,
                stop_early=stop_early, abs_tol=abs_tol, rel_tol=rel_tol,
                residual_term=self.residual_term
            )
            result['it'] = len(result['obj_vals']) - 1
            self.admm_result = result
            if verbose:
                print('\npolishing...\n')
            result = run_bcd(
                self.data, self.classes, num_iter=num_iter, use_ix=use_set,
                abs_tol=abs_tol, rel_tol=rel_tol, X_init=result['X'],
                verbose=verbose
            )
            self.bcd_result = result
            for key in ['obj_vals', 'optimality_residual']:
                self.admm_result[key] = np.r_[
                    self.admm_result[key], self.bcd_result[key]
                ]
            self.components = result['X']
        elif not self.is_convex and how.lower() in ['cvx', 'cvxpy']:
            m1 = 'This problem is non-convex and not solvable with CVXPY. '
            m2 = 'Please try solving with ADMM.'
            print(m1 + m2)
        else:
            m1 = "Sorry, I didn't catch that. Please select the 'how' kwarg \n"
            m2 = "from 'cvxpy', 'bcd', 'admm', or 'admm-polish'. "
            print(m1 + m2)

    def set_weights(self, weights):
        if len(self.__weights.value) == len(weights):
            self.__weights.value = weights
        elif len(self.__weights.value) == len(weights) + 1:
            self.__weights.value = np.r_[[1], weights]
        for c, w in zip(self.classes, self.weights):
            c.set_weight(w)
        return


    def holdout_validation(self, holdout=0.2, seed=None, rho=None,
                           rho0_scale=None, how=None, num_iter=1e3,
                           verbose=True, reset=True, X_init=None, u_init=None,
                           stop_early=True, abs_tol=1e-5, rel_tol=1e-5,
                           **cvx_kwargs):
        if seed is not None:
            np.random.seed(seed)
        size = self.T * self.p
        if self.p == 1:
            known_ixs = np.arange(size)[self.known_set]
        else:
            known_ixs = np.arange(size)[self.known_set.ravel(order='F')]
        train_ixs, test_ixs = train_test_split(
            known_ixs, test_size=holdout, random_state=seed
        )

        hold_set = np.zeros(size, dtype=bool)
        use_set = np.zeros(size, dtype=bool)
        hold_set[test_ixs] = True
        use_set[train_ixs] = True
        if self.p != 1:
            hold_set = hold_set.reshape((self.T, self.p), order='F')
            use_set = use_set.reshape((self.T, self.p), order='F')
        self.decompose(use_set=use_set, rho=rho, rho0_scale=rho0_scale,
                       how=how, num_iter=num_iter, verbose=verbose,
                       reset=reset, X_init=X_init, u_init=u_init,
                       stop_early=stop_early, abs_tol=abs_tol, rel_tol=rel_tol,
                       **cvx_kwargs)
        y_hat = np.sum(self.components[:, hold_set], axis=0)
        hold_y = self.data[hold_set]
        residuals = hold_y - y_hat
        holdout_cost = np.average(np.power(residuals, 2))
        return holdout_cost

    def plot_decomposition(self, x_series=None, X_real=None, figsize=(10, 8),
                           label='estimated', exponentiate=False,
                           skip=None, **kwargs):
        if self.components is None:
            print('No decomposition available.')
            return
        if not exponentiate:
            f = lambda x: x
            base = 'Component $x'
        else:
            f = lambda x: np.exp(x)
            base = 'Component $\\tilde{x}'
        if skip is not None:
            skip = np.atleast_1d(skip)
            nd = len(skip)
        else:
            nd = 0
        K = len(self.classes)
        fig, ax = plt.subplots(nrows=K + 1 - nd, sharex=True, figsize=figsize, **kwargs)
        if x_series is None:
            xs = np.arange(self.components.shape[1])
        else:
            xs = np.copy(x_series)
        ax_ix = 0
        for k in range(K + 1):
            if skip is not None and k in skip:
                continue
            if k == 0:
                est = self.components[k]
                ax[ax_ix].plot(xs, f(est), label=label, linewidth=1,
                               ls='none', marker='.', ms=2)
                ax[ax_ix].set_title(base + '^{}$'.format(k + 1))
                if X_real is not None:
                    true = X_real[k]
                    ax[ax_ix].plot(true, label='true', linewidth=1)
            elif k < K:
                est = self.components[k]
                ax[ax_ix].plot(xs, f(est), label=label, linewidth=1)
                ax[ax_ix].set_title(base + '^{}$'.format(k + 1))
                if X_real is not None:
                    true = X_real[k]
                    ax[ax_ix].plot(xs, true, label='true', linewidth=1)
            else:
                if not exponentiate:
                    lbl = 'observed, $y$'
                else:
                    lbl = 'observed, $\\tilde{y}$'
                ax[ax_ix].plot(xs, f(self.data), label=lbl,
                           linewidth=1, color='green')
                ax[ax_ix].plot(xs, f(np.sum(self.components[1:], axis=0)),
                               label='denoised estimate', linewidth=1)
                if X_real is not None:
                    ax[ax_ix].plot(xs, np.sum(X_real[1:], axis=0), label='true',
                               linewidth=1)
                ax[ax_ix].set_title('composed signal')
                ax[ax_ix].legend()
            if X_real is not None:
                ax[ax_ix].legend()
            ax_ix += 1
        plt.tight_layout()
        return fig

    # def optimize_weights(self, solver='ECOS', seed=None):
    #     if seed is None:
    #         seed = np.random.random_integers(0, 1000)
    #     if self.num_components == 2:
    #         search_ix = 1 - self.residual_term
    #         _ = self.holdout_validation(solver=solver, seed=seed)
    #         new_vals = np.ones(2)
    #
    #         def cost_meta(v):
    #             val = 10 ** v
    #             new_vals[search_ix] = val
    #             self.weights.value = new_vals
    #             cost = self.holdout_validation(solver=solver, seed=seed,
    #                                            reuse=True)
    #             return cost
    #         res = minimize_scalar(cost_meta, bounds=(-2, 10), method='bounded')
    #         best_val = 10 ** res.x
    #         new_vals[search_ix] = best_val
    #         self.weights.value = new_vals
    #         self.demix(solver=solver, reset=True)
    #         return
    #     else:
    #         print('IN PROGRESS')
    #
    # def optimize_parameters(self, solver='ECOS', seed=None):
    #     if seed is None:
    #         seed = np.random.random_integers(0, 1000)
    #     if self.num_parameters == 1:
    #         k1, k2 = [(k1, k2) for k1, value in self.parameters.items()
    #                   for k2 in value.keys()][0]
    #         _ = self.holdout_validation(solver=solver, seed=seed)
    #         def cost_meta(val):
    #             self.parameters[k1][k2].value = val
    #             cost = self.holdout_validation(solver=solver, seed=seed,
    #                                            reuse=True)
    #             return cost
    #         res = minimize_scalar(cost_meta, bounds=(0, 1), method='bounded')
    #         best_val = res.x
    #         self.parameters[k1][k2].value = best_val
    #         self.decompose(solver=solver, reset=True)
    #         return
    #     else:
    #         print('IN PROGRESS')

    def __construct_cvx_problem(self, use_set=None):
        if use_set is None:
            use_set = self.known_set
        self.use_set = use_set
        if len(self.data.shape) == 1:
            p = 1
        else:
            p = self.data.shape[1]
        y_tilde = np.copy(self.data)
        y_tilde[np.isnan(y_tilde)] = 0
        T = self.T
        K = self.K
        weights = self.__weights
        if p == 1:
            xs = [cvx.Variable(T, name='x_{}'.format(i)) for i in range(K)]
        else:
            xs = [cvx.Variable((T, p), name='x_{}'.format(i)) for i in range(K)]
        costs = [c.cost(x) for c, x in zip(self.classes, xs)]
        costs = [weights[i] * cost for i, cost in enumerate(costs)]
        # print([c.is_dcp() for c in costs])
        # print([c.sign for c in costs])
        # print([c.curvature for c in costs])
        # print(cvx.sum(costs).is_dcp())
        constraints = [
            c.make_constraints(x) for c, x in zip(self.classes, xs)
        ]
        constraints = list(chain.from_iterable(constraints))
        constraints.append(cvx.sum([x for x in xs], axis=0)[use_set]
                           == y_tilde[use_set])
        objective = cvx.Minimize(cvx.sum(costs))
        problem = cvx.Problem(objective, constraints)
        self._problem_costs = costs
        # print(problem.is_dcp())
        return problem

class GraphProblem():
    def __init__(self, data, components):
        self.data = data
        self.components = components
