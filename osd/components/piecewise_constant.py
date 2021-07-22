''' Piecewise Constant Signal

This module contains the class for a piecewise constant signal with a known
number of jumps. This class implements a dynamic programming algorithm for
optimal segmentation of a scalar signal into a given number of separate linear
models. See this paper for more details:

A. Kehagias, E. Nidelkou, and V. Petridis
"A dynamic programming segmentation procedure for hydrological and
environmental time series"
https://link.springer.com/article/10.1007/s00477-005-0013-6

Author: Bennet Meyers
'''

from scipy import sparse
import numpy as np
from osd.components.component import Component

class PiecewiseConstant(Component):

    def __init__(self, num_segments=2, **kwargs):
        super().__init__(**kwargs)
        self.num_segments = num_segments
        return

    @property
    def is_convex(self):
        return False

    def _get_cost(self):
        return lambda x: 0

    def prox_op(self, v, weight, rho):
        d = error(v)
        x = dp_seg(v, d, self.num_segments)
        return x


def error(x):
    """
    Calculate the "error matrix" for the piecewise constant partition problem.
    This is simply a matrix whose entries (i, j) contains the scaled variance
    of the signal segment x[i:j] for i < j.

    :param x: 1D numpy array containing signal to be partitioned
    :return:  2D numpy array containing error matrix
    """
    T = len(x)
    d = sparse.lil_matrix((T, T), dtype=np.float32)
    for a in range(T):
        A_top = np.cumsum(x[a:])
        A_bot = np.arange(1, T - a + 1)
        A_row = np.divide(A_top, A_bot)
        SS_row = np.divide(np.cumsum(np.power(x[a:], 2)), A_bot)
        d[a, a:] = np.multiply(SS_row - np.power(A_row, 2), A_bot)
    return d

def dp_seg(x, d, c):
    """
    Dynamic programming algorithm for optimal partitioning of a signal into
    piecewise constant segments, based on finding the partition with the
    minimal residual variance.

    :param x: time series x_1, ... , x_T
    :param d: precalculated segment errors
    :param c: the number of segments, typically c >= 2
    :return:
    """
    T = len(x)
    # Cost table
    cost = np.zeros((c, T))
    cost[:] = np.nan
    cost[0, :] = d[0, :].toarray() # *one* segment (first index) has *zero* jumps
    # Index table
    z = np.zeros((c, T))
    z[:] = np.nan
    z[0, :] = np.arange(T)
    # Minimization
    for k in range(1, c):
        e = np.zeros((T, T))
        e[:] = np.nan
        for b_right in range(T):
            if b_right >= k:
                e_entry = [cost[k - 1, b_left] + d[b_left + 1, b_right]
                           if b_left >= k - 1
                           else np.nan
                           for b_left in range(b_right)]
                e[b_right, :b_right] = e_entry
        slct = ~np.alltrue(np.isnan(e), axis=-1)
        cost[k, slct] = np.nanmin(e[slct, :], axis=-1)
        z[k, slct] = np.nanargmin(e[slct, :], axis=-1) + 1 # the optimal b_right
    # Backtracking
    # Change points are designated by the last index of the segment.
    # So, the last change point is always the final index
    segs = np.eye(c) * (T)
    for k1 in range(c):
        for k2 in range(1, k1+1)[::-1]:
            segs[k1, k2-1] = z[k2, int(segs[k1, k2] - 1)]
    estimate = np.ones_like(x)
    bps = np.r_[[0], segs[-1]]
    for i in range(len(bps) - 1):
        a = int(bps[i])
        b = int(bps[i + 1])
        estimate[a:b] = np.average(x[a:b])
    return estimate

def calc_cost(x, breakpoints):
    cost = 0
    bp = breakpoints
    if bp[0] != 0:
        bp = np.r_[[0], bp]
    for i in range(len(bp) - 1):
        a = int(bp[i])
        b = int(bp[i + 1])
        cost += (b - a) * np.var(x[a:b])
    return cost