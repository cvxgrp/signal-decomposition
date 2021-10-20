''' One-Jump Component

This module contains the class for a signal that is allowed to have exactly one
change in value or "jump". The jump is chosen to minimize the total variance of
the two subsections of the signal. The cost function is

    phi(x) = 1 if there is a jump and 0 otherwise

The component cost can take a weight parameter, which provides control over the
size of the jump that will be detected by the component, both in terms of the
absolute value of the change and the number of points included in change.

Author: Bennet Meyers
'''

import pandas as pd
import numpy as np
from osd.components.component import Component

class OneJump(Component):
    def __init__(self, start_value=None, end_value=None, min_fraction=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.start_value = start_value
        self.end_value = end_value # TODO: add this to prox op
        self.min_fraction = min_fraction
        return

    @property
    def is_convex(self):
        return False

    def _get_cost(self):
        f = lambda x: 0 if x[-1] == x[0] else 1
        return f

    def prox_op(self, v, weight, rho):
        if self.start_value is None:
            mu = np.average(v)
        else:
            mu = self.start_value
        cost_no_jump = (rho / 2) * np.sum(np.power(v - mu, 2))
        results, best_ix = find_jump(v, min_fraction=self.min_fraction)
        x_jump = np.ones_like(v)
        if self.start_value is None:
            x_jump[:best_ix] = results.loc[best_ix]['mu1']
        else:
            x_jump[:best_ix] = start
        x_jump[best_ix:] = results.loc[best_ix]['mu2']
        cost_with_jump = weight + (rho / 2) * np.sum(np.power(v - x_jump, 2))
        if cost_with_jump < cost_no_jump:
            # print('jump!')
            return x_jump
        else:
            # print('no jump!')
            return mu * np.ones_like(v)


def find_jump(signal, min_fraction=None):
    """
    Find the breakpoint in a scalar signal that minimizes the total variance in
    both signal segments
    :param signal: a scalar signal (typically 1D numpy array)
    :return: dataframe with results of search and optimal breakpoint index (tuple)
    """
    N = len(signal)
    cum_sum_left = np.r_[[0], np.cumsum(signal)]
    cum_sum_right = np.r_[np.cumsum(signal[::-1])[::-1], [0]]
    cum_sum_sq_left = np.r_[[0], np.cumsum(np.power(signal, 2))]
    cum_sum_sq_right = np.r_[np.cumsum(np.power(signal[::-1], 2))[::-1], [0]]
    denoms = np.arange(N + 1)
    mu_left = np.nan * np.ones_like(cum_sum_left)
    mu_right = np.nan * np.ones_like(cum_sum_right)
    var_left = np.zeros_like(cum_sum_sq_left)
    var_right = np.zeros_like(cum_sum_sq_right)
    np.divide(cum_sum_left, denoms, out=mu_left, where=denoms != 0)
    np.divide(cum_sum_right, denoms[::-1], out=mu_right,
              where=denoms[::-1] != 0)
    np.divide(cum_sum_sq_left, denoms, out=var_left, where=denoms != 0)
    np.divide(cum_sum_sq_right, denoms[::-1],
              out=var_right, where=denoms[::-1] != 0)
    vls = ~np.isnan(mu_left)
    var_left[vls] -= np.power(mu_left[vls], 2)
    vrs = ~np.isnan(mu_right)
    var_right[vrs] -= np.power(mu_right[vrs], 2)
    results = pd.DataFrame(data={
        'mu1': mu_left, 'var1': var_left, 'mu2': mu_right, 'var2': var_right
    })
    results['total_var'] = results['var1'] + results['var2']
    if min_fraction is None:
        best_ix = np.argmin(results['total_var'])
    else:
        view = results[results['fraction'] >= min_fraction]
        best_ix = view.index[np.argmin(view['total_var'].values)]
    return results, best_ix