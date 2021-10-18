""" Block Coordinate Descent (BCD) for signal decomposition

Author: Bennet Meyers
"""

import numpy as np
from osd.signal_decomp_admm import calc_obj

def run_bcd(data, components, num_iter=50, use_ix=None):
    if use_ix is None:
        use_ix = np.ones_like(data, dtype=bool)
    y = data
    T = len(data)
    K = len(components)
    rho = 2 / T
    use_ix = np.ones_like(y, dtype=bool)
    X = np.zeros((K, T))
    X[0, use_ix] = y[use_ix]
    X0_next = np.copy(X[0, :])
    obj = np.zeros(num_iter * 2 + 1)
    obj[0] = calc_obj(y, X, components, use_ix, residual_term=0)
    gradients = np.zeros_like(X)
    norm_dual_residual = np.zeros_like(obj)
    counter = 1
    for it in range(num_iter):
        for k in range(1, K):
            prox = components[k].prox_op
            weight = components[k].weight
            #### Coordinate descent updates
            Xk_next = prox(X[0, :] + X[k, :], weight, rho)
            X0_next[use_ix] = (X[0, :] + X[k, :] - Xk_next)[use_ix]
            gradients[k, :] = rho * (X[0, :] + X[k, :] - Xk_next)
            X[0, :] = X0_next
            X[k, :] = Xk_next
            gradients[0] = X[0] * 2 / y.size
            dual_resid = gradients - X[0] * 2 / y.size
            n_s_k = np.linalg.norm(dual_resid)
            obj[counter] = calc_obj(y, X, components, use_ix,
                                    residual_term=0)
            norm_dual_residual[counter] = n_s_k
            counter += 1
    out_dict = {
        'X': X,
        'obj_vals': obj,
        'dual_r': norm_dual_residual
    }
    return out_dict