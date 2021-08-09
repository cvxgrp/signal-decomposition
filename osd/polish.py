''' Polishing module

This module contains functions for polishing non-convex solutions

Author: Bennet Meyers
'''

import numpy as np
from osd.signal_decomp_admm import make_estimate

def boolean_polish(problem, boolean_component, scale, residual_component=0):
    swapped_ix = []
    obj_val = problem.objective_value
    for ix in range(problem.T):
        est = problem.estimates[boolean_component, ix]
        if np.isclose(est, 0):
            problem.estimates[boolean_component, ix] = scale
        elif np.isclose(est, scale):
            problem.estimates[boolean_component, ix] = 0
        if problem.objective_value < obj_val:
            obj_val = problem.objective_value
            problem.estimates = make_estimate(problem.data, problem.estimates,
                                              problem.use_set)
            swapped_ix.append(ix)
        else:
            problem.estimates[boolean_component, ix] = est
    return swapped_ix