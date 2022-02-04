""" Block Coordinate Descent (BCD) for signal decomposition

Author: Bennet Meyers
"""

import numpy as np
from time import time
from osd.masking import Mask
from osd.utilities import calc_obj, make_estimate, AlgProgress


def run_bcd(data, components, num_iter=50, use_ix=None, X_init=None,
            abs_tol=1e-5, rel_tol=1e-5, rho=None, verbose=True, debug=False):
    if use_ix is None:
        use_ix = ~np.isnan(data)
    else:
        use_ix = np.logical_and(use_ix, ~np.isnan(data))
    mask_op = Mask(use_ix)
    y = data
    if len(data.shape) == 1:
        T = len(data)
        p = 1
    else:
        T, p = data.shape
    K = len(components)
    if rho is None:
        rho = 2 / (T * p)
    if X_init is None:
        if p == 1:
            X = np.zeros((K, T))
        else:
            X = np.zeros((K, T, p))
        X[0, use_ix] = y[use_ix]
    else:
        X = np.copy(X_init)
    indices = np.arange(K)
    X0_next = np.copy(X[0, :])
    obj = []
    # obj.append(calc_obj(y, X, classes, use_ix, residual_term=0))
    gradients = np.zeros_like(X)
    residual = []
    if verbose:
        m1 = 'Starting BCD...\n'
        m1 += 'y shape: {}\n'.format(y.shape)
        m1 += 'X shape: {}\n'.format(X.shape)
        print(m1)
    ti = time()
    prog = AlgProgress(num_iter, ti)
    for it in range(num_iter):
        for k in range(1, K):
            prox = components[k].prox_op
            weight = components[k].weight
            #### Coordinate descent updates
            rhs = np.sum(X[np.logical_and(indices != 0, indices !=k)], axis=0)
            vin = data - rhs
            vout = prox(vin, weight, rho, use_set=use_ix)
            gradients[k, :] = rho * mask_op.zero_fill(vin - vout)
            X[k, :] = vout
            if debug:
                info = (it, k, calc_obj(y, X,components,use_ix))
                print('it: {}; k: {}, obj_val: {:.3e}'.format(*info))
        X = make_estimate(data, X, use_ix)
        gradients[0] = X[0] * 2 / y.size
        r = np.sqrt(
            (1 / (K - 1)) * np.sum(np.power(
                gradients[indices !=  0] - gradients[0], 2))
        )
        obj_val = calc_obj(y, X, components, use_ix, residual_term=0)
        obj.append(obj_val)
        residual.append(r)
        stopping_tolerance = abs_tol + rel_tol * np.linalg.norm(gradients[0])
        if verbose:
            prog.print(obj_val, r, stopping_tolerance)
        if r <= stopping_tolerance and it >= 1:
            if verbose:
                prog.print(obj_val, r, stopping_tolerance, done=True)
            break
    out_dict = {
        'X': X,
        'obj_vals': obj,
        'optimality_residual': residual
    }
    return out_dict