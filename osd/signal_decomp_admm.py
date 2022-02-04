''' ADMM for Signal Decomposition

This module contains an implementation of the ADMM algorithm for Signal
Decomposition presented in the paper "Signal Decomposition via Distributed
Optimization"

Author: Bennet Meyers
'''

import numpy as np
from time import time
from osd.masking import Mask
from osd.utilities import make_estimate, calc_obj, AlgProgress
import matplotlib.pyplot as plt


def run_admm(data, components, num_iter=50, rho=None, use_ix=None, verbose=True,
             X_init=None, u_init=None, stop_early=True, residual_term=0,
             abs_tol=1e-5, rel_tol=1e-5, debug=False):
    """
    Serial implementation of SD ADMM algorithm.

    :param data: numpy array containing problem data
    :param components: list of osd.classes class objects
    :param num_iter: (int) the number of ADMM iterations to perform
    :param rho: (float) the ADMM learning rate
    :param use_ix: (None or Boolean array) the set of known index values
    :param verbose: (Boolean) print progress to screen
    :param randomize_start: (Boolean) Randomize initialization of classes
    :return:
    """
    y = data
    if use_ix is None:
        use_ix = ~np.isnan(data)
    else:
        use_ix = np.logical_and(use_ix, ~np.isnan(data))
    mask_op = Mask(use_ix)
    if len(data.shape) == 1:
        T = len(data)
        p = 1
    else:
        T, p = data.shape
    K = len(components)
    indices = np.arange(K)
    if X_init is None:
        if p == 1:
            X = np.zeros((K, T))
        else:
            X = np.zeros((K, T, p))
        X[0, use_ix] = y[use_ix]
    elif p == 1 and X_init.shape == (K, T):
        X = np.copy(X_init)
    elif p > 1 and X_init.shape == (K, T, p):
        X = np.copy(X_init)
    else:
        m1 = 'A initial value was given for X that does not match the problem shape.'
        print(m1)
        return
    if u_init is None:
        u = np.zeros(mask_op.q)
    elif u_init.shape == (mask_op.q, ):
        u = np.copy(u_init)
    else:
        m1 = 'A initial value was given for u that does not match the problem known set.'
        print(m1)
        return
    gradients = np.zeros_like(X)
    residual = []
    obj_vals = []
    ti = time()
    best = {
        'X': None,
        'u': None,
        'it': None,
        'obj_val': np.inf
    }
    if rho is None:
        rho = 2 / (T * p)
    if len(np.atleast_1d(rho)) == 1:
        rho = np.ones(num_iter, dtype=float) * rho
    else:
        num_iter = len(rho)
    prog = AlgProgress(num_iter, ti)
    if verbose:
        m1 =  'Starting ADMM...\n'
        m1 += 'y shape: {}\n'.format(y.shape)
        m1 += 'X shape: {}\n'.format(X.shape)
        m1 += 'u shape: {}\n'.format(u.shape)
    for it, rh in enumerate(rho):
        # Apply proximal operators for each signal class
        for k in range(K):
            prox = components[k].prox_op
            weight = components[k].weight
            vin = X[k, :] - 2 * mask_op.unmask(u)
            x_new = prox(vin, weight, rh, use_set=use_ix)
            if debug:
                plt.plot(vin, label='vin')
                plt.plot(x_new, label='vout')
                plt.legend()
                plt.title('Comp {}, iteration {}'.format(k + 1, it + 1))
                plt.show()
            gradients[k, :] = rh * mask_op.zero_fill(vin - x_new)
            X[k, :] = x_new

        # dual_update
        u += mask_op.mask(np.average(X, axis=0) - y / K)
        # record keeping
        obj_val = calc_obj(y, X, components, use_ix,
                           residual_term=residual_term)
        obj_vals.append(obj_val)
        X_tilde = make_estimate(y, X, use_ix, residual_term=residual_term)
        gradients[0] = X_tilde[0] * 2 / y.size
        r = np.sqrt(
            (1 / (K - 1)) * np.sum(np.power(
                gradients[indices != 0] - gradients[0], 2))
        )
        residual.append(r)
        stopping_tolerance = abs_tol + rel_tol * np.linalg.norm(gradients[0])
        if verbose:
            prog.print(obj_val, r, stopping_tolerance)
        if (obj_val < best['obj_val'] and stop_early):
            best = {
                'X': X_tilde,
                'u': u,
                'it': it,
                'obj_val': obj_val
            }
        if r <= stopping_tolerance or (stop_early and it - best['it'] > 20):
            if verbose:
                prog.print(obj_val, r, stopping_tolerance, done=True)
            break
    if best['obj_val'] == np.inf:
        stop_early = False
    if not stop_early:
        X_tilde = make_estimate(y, X, use_ix)
        best = {
            'X': X_tilde,
            'u': u,
            'it': it,
            'obj_val': obj_val
        }
    outdict = {
        'X': best['X'],
        'u': best['u'],
        'it': best['it'],
        'optimality_residual': residual,
        'obj_vals': obj_vals,
        'best_obj': best['obj_val']
    }
    return outdict