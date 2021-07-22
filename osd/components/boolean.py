''' Boolean Signal

This module contains the class for Boolean signal


Author: Bennet Meyers
'''

import numpy as np
from osd.components.component import Component

class Boolean(Component):

    def __init__(self, scale=1, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
        return

    @property
    def is_convex(self):
        return False

    def _get_cost(self):
        return lambda x: 0

    def prox_op(self, v, weight, rho):
        r_0 = np.abs(v)
        r_1 = np.abs(v - self.scale)
        x = np.zeros_like(v)
        x[r_1 <= r_0] = self.scale
        return x