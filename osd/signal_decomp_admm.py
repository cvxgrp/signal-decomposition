''' ADMM for Signal Decomposition

This module contains an implementation of the ADMM algorithm for Signal
Decomposition presented in the paper "Signal Decomposition via Distributed
Optimization"

Author: Bennet Meyers
'''

import numpy as np
from time import time
from osd.utilities import progress, make_estimate, calc_obj


def run_admm(data, components, num_iter=50, rho=1., use_ix=None, verbose=True,
             randomize_start=False, X_init=None, u_init=None, stop_early=False,
             residual_term=0, stopping_tolerance=1e-6):
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
    if len(data.shape) == 1:
        T = len(data)
        p = 1
    else:
        T, p = data.shape
    K = len(components)
    if use_ix is None:
        use_ix = np.ones_like(data, dtype=bool)
    if X_init is None:
        if p == 1:
            X = np.zeros((K, T))
        else:
            X = np.zeros((K, T, p))
        if not randomize_start:
            X[0, use_ix] = y[use_ix]
        else:
            if p == 1:
                X[1:, :] = np.random.randn(K-1, T)
            else:
                X[1:, :] = np.random.randn(K - 1, T, p)
            X[0, use_ix] = y[use_ix] - np.sum(X[1:, use_ix], axis=0)
    elif p == 1 and X_init.shape == (K, T):
        X = np.copy(X_init)
    elif p > 1 and X_init.shape == (K, T, p):
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
        X_tilde = make_estimate(y, X, use_ix, residual_term=residual_term)
        dual_resid = gradients[1:] - X_tilde[0] * 2 / (components[0].size *
                                                       components[0].weight)
        dual_resid = dual_resid[:, use_ix]
        # n_s_k = np.linalg.norm(dual_resid) / np.sqrt(dual_resid.size)
        n_s_k = np.sum(np.power(dual_resid, 2)) / (K - 1)
        norm_dual_residual.append(n_s_k)
        obj_val = calc_obj(y, X, components, use_ix,
                           residual_term=residual_term)
        obj_vals.append(obj_val)
        if (obj_val < best['obj_val'] and n_s_k <= stopping_tolerance
                and stop_early):
            best = {
                'X': X_tilde,
                'u': u,
                'it': it,
                'obj_val': obj_val
            }
        if np.average(norm_dual_residual[-5:]) <= stopping_tolerance:
            break
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
        progress(num_iter, num_iter, '{:.2f} sec\n'.format(td))
    outdict = {
        'X': best['X'],
        'u': best['u'],
        'it': best['it'],
        'optimality_residual': norm_dual_residual,
        'obj_vals': obj_vals,
        'best_obj': best['obj_val']
    }
    return outdict