
import numpy as np
import scipy.sparse as sp
from gfosd.components.base_graph_class import GraphComponent


class Inequality(GraphComponent):
    def __init__(self, vmax=None, vmin=None, *args, **kwargs):
        self._max = vmax
        self._min = vmin
        if self._max is not None and self._min is not None:
            self._kind = 'box'
        elif self._max is not None:
            self._kind = 'max'
        elif self._min is not None:
            self._kind = 'min'
        else:
            print('Please enter a max value and/or a min value')
            raise Exception
        super().__init__(weight=1, *args, **kwargs)

    def _make_gz(self):
        if self._kind == 'box':
            self._gz = [{'g': 'box',
                         'args': None,
                         'range': (0, self.z_size)}]
        elif self._kind == 'max':
            self._gz = [{'g': 'nonpos',
                         'args': None,
                         'range': (0, self.z_size)}]
        elif self._kind == 'min':
            self._gz = [{'g': 'nonneg',
                         'args': None,
                         'range': (0, self.z_size)}]

    def _make_B(self):
        if self._kind == 'box':
            self._B = -1 * (self._max - self._min) * sp.eye(self.z_size)
        elif self._kind == 'max' or self._kind == 'min':
            self._B = -1 * sp.eye(self.z_size)

    def _make_c(self):
        if self._kind == 'box':
            self._c = self._min * np.ones(self.x_size)
        elif self._kind == 'max':
            self._c = self._max * np.ones(self.x_size)
        elif self._kind == 'min':
            self._c = self._min * np.ones(self.x_size)

