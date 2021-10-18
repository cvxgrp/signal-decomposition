''' ADMM for Signal Decomposition

This module contains an implementation of the ADMM algorithm for Signal
Decomposition presented in the paper "Signal Decomposition via Distributed
Optimization"

Author: Bennet Meyers
'''

import numpy as np
from time import time
from osd.utilities import progress

def make_estimate(y, X, use_ix, residual_term=0):
    """
    After any given iteration of the ADMM algorithm, generate an estimate that
    is feasible with respect to the global equality constraint by making x0
    equal to the residual between the input data y and the rest of the
    component estimates

    :param y: numpy array containing problem data
    :param X: current estimate of decomposed signal components from ADMM
    :param use_ix: the known index set (Boolean array)
    :return: the estimate with the first component replaced by the residuals
    """
    X_tilde = np.copy(X)
    sum_ix = np.arange(X.shape[0])
    sum_ix = np.delete(sum_ix, residual_term)
    X_tilde[residual_term, use_ix] = y[use_ix] - np.sum(X[sum_ix][:, use_ix],
                                                        axis=0)
    X_tilde[residual_term, ~use_ix] = 0
    return X_tilde

def calc_obj(y, X, components, use_ix, residual_term=0):
    """
    Calculate the current objective value of the problem

    :param y: numpy array containing problem data
    :param X: current estimate of decomposed signal components from ADMM
    :param use_ix: the known index set (Boolean array)
    :return: the scalar problem objective value
    """
    K = len(components)
    X_tilde = make_estimate(y, X, use_ix, residual_term=residual_term)
    obj_val = 0
    for k in range(K):
        try:
            cost = components[k].cost(X_tilde[k]).value.item()
        except AttributeError:
            cost = components[k].cost(X_tilde[k])
        weight = components[k].weight
        obj_val += weight * cost
    return obj_val

def run_admm(data, components, num_iter=50, rho=1., use_ix=None, verbose=True,
             randomize_start=False, X_init=None, u_init=None, stop_early=False,
             residual_term=0):
    """
    Serial implementation of SD ADMM algorithm.

    :param data: numpy array containing problem data
    :param components: list of osd.components class objects
    :param num_iter: (int) the number of ADMM iterations to perform
    :param rho: (float) the ADMM learning rate
    :param use_ix: (None or Boolean array) the set of known index values
    :param verbose: (Boolean) print progress to screen
    :param randomize_start: (Boolean) Randomize initialization of components
    :return:
    """
    y = data
    T = len(data)
    K = len(components)
    if use_ix is None:
        use_ix = np.ones_like(data, dtype=bool)
    if X_init is None:
        X = np.zeros((K, T))
        if not randomize_start:
            X[0, use_ix] = y[use_ix]
        else:
            X[1:, :] = np.random.randn(K-1, T)
            X[0, use_ix] = y[use_ix] - np.sum(X[1:, use_ix], axis=0)
    elif X_init.shape == (K, T):
        X = np.copy(X_init)
    else:
        m1 = 'A initial value was given for X that does not match the problem shape.'
        print(m1)
        return
    if u_init is None:
        u = np.zeros_like(y)
    else:
        u = np.copy(u_init)
    gradients = np.zeros_like(X)
    norm_primal_residual = []
    norm_dual_residual = []
    obj_vals = []
    ti = time()
    best = {
        'X': None,
        'u': None,
        'it': None,
        'obj_val': np.inf
    }
    if len(np.atleast_1d(rho)) == 1:
        rho = np.ones(num_iter, dtype=float) * rho
    else:
        num_iter = len(rho)
    for it, rh in enumerate(rho):
        if verbose:
            td = time() - ti
            progress(it, num_iter, '{:.2f} sec'.format(td))
        # Apply proximal operators for each signal class
        for k in range(K):
            prox = components[k].prox_op
            weight = components[k].weight
            x_new = prox(X[k, :] - u, weight, rh)
            gradients[k, :] = rh * (X[k, :] - u - x_new)
            X[k, :] = x_new
        # Consensus step
        u[use_ix] += 2 * (np.average(X[:, use_ix], axis=0) - y[use_ix] / K)
        # calculate primal and dual residuals
        primal_resid = np.sum(X, axis=0)[use_ix] - y[use_ix]
        dual_resid = gradients - X[0] * 2 / y.size
        n_r_k = np.linalg.norm(primal_resid)
        n_s_k = np.linalg.norm(dual_resid)
        norm_primal_residual.append(n_r_k)
        norm_dual_residual.append(n_s_k)

        obj_val = calc_obj(y, X, components, use_ix,
                           residual_term=residual_term)
        obj_vals.append(obj_val)
        if obj_val < best['obj_val'] and stop_early:
            X_tilde = make_estimate(y, X, use_ix, residual_term=residual_term)
            best = {
                'X': X_tilde,
                'u': u,
                'it': it,
                'obj_val': obj_val
            }
    if not stop_early:
        X_tilde = make_estimate(y, X, use_ix)
        best = {
            'X': X_tilde,
            'u': u,
            'it': it,
            'obj_val': obj_val
        }
    if verbose:
        td = time() - ti
        progress(it + 1, num_iter, '{:.2f} sec\n'.format(td))
    outdict = {
        'X': best['X'],
        'u': best['u'],
        'it': best['it'],
        'primal_r': norm_primal_residual,
        'dual_r': norm_dual_residual,
        'obj_vals': obj_vals,
        'best_obj': best['obj_val']
    }
    return outdict