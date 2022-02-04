''' Polishing module

This module contains functions for polishing non-convex solutions

Author: Bennet Meyers
'''

import numpy as np
from osd.utilities import make_estimate


def boolean_polish(problem, boolean_component, scale, residual_term=0):
    swapped_ix = []
    obj_val = problem.objective_value
    for ix in range(problem.T):
        est = problem.components[boolean_component, ix]
        if np.isclose(est, 0):
            problem.components[boolean_component, ix] = scale
        elif np.isclose(est, scale):
            problem.components[boolean_component, ix] = 0
        if problem.objective_value < obj_val:
            obj_val = problem.objective_value
            problem.components = make_estimate(problem.data, problem.components,
                                               problem.use_set,
                                               residual_term=residual_term)
            swapped_ix.append(ix)
        else:
            problem.components[boolean_component, ix] = est
    return swapped_ix