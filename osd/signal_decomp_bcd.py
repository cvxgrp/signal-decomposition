""" Block Coordinate Descent (BCD) for signal decomposition

Author: Bennet Meyers
"""

import numpy as np
from time import time
from osd.masking import Mask
from osd.utilities import calc_obj, progress, make_estimate


def run_bcd(data, components, num_iter=50, use_ix=None, X_init=None,
            abs_tol=1e-5, rel_tol=1e-5, rho=None, verbose=True):
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
    # obj.append(calc_obj(y, X, components, use_ix, residual_term=0))
    gradients = np.zeros_like(X)
    residual = []
    ti = time()
    for it in range(num_iter):
        if verbose:
            td = time() - ti
            if td < 60:
                progress(it, num_iter, '{:.2f} sec   '.format(td))
            else:
                progress(it, num_iter, '{:.2f} min   '.format(td / 60))
        for k in range(1, K):
            prox = components[k].prox_op
            weight = components[k].weight
            #### Coordinate descent updates
            rhs = np.sum(X[np.logical_and(indices != 0, indices !=k)])
            vin = data - rhs
            vout = prox(vin, weight, rho, use_set=use_ix)
            gradients[k, :] = rho * mask_op.zero_fill(vin - vout)
            X[k, :] = vout
        X = make_estimate(data, X, use_ix)
        gradients[0] = X[0] * 2 / y.size
        r = np.sqrt(
            (1 / (K - 1)) * np.sum(np.power(
                gradients[indices !=  0] - gradients[0], 2))
        )
        obj.append(calc_obj(y, X, components, use_ix,
                                residual_term=0))
        residual.append(r)
        stopping_tolerance = abs_tol + rel_tol * np.linalg.norm(gradients[0])
        if r <= stopping_tolerance:
            break
    if verbose:
        td = time() - ti
        progress(num_iter, num_iter, '{:.2f} sec\n'.format(td))
    out_dict = {
        'X': X,
        'obj_vals': obj,
        'optimality_residual': residual
    }
    return out_dict