''' Boolean Signal

This module contains the class for Boolean signal


Author: Bennet Meyers
'''

import numpy as np
from osd.classes.component import Component

class Boolean(Component):

    def __init__(self, scale=1, shift=0, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
        self.shift = shift
        return

    @property
    def is_convex(self):
        return False

    def _get_cost(self):
        return lambda x: 0

    def prox_op(self, v, weight, rho, use_set=None, prox_weights=None):
        low_val = self.shift
        high_val = self.scale + self.shift
        r_0 = np.abs(v - low_val)
        r_1 = np.abs(v - high_val)
        x = np.ones_like(v) * low_val
        x[r_1 <= r_0] = high_val
        # deterministic behavior when there are missing values
        if use_set is not None:
            x[~use_set] = low_val
        return x