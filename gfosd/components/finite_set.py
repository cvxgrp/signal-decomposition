
import numpy as np
import scipy.sparse as sp
from gfosd.components.base_graph_class import GraphComponent

class FiniteSet(GraphComponent):
    def __init__(self, set=None, *args, **kwargs):
        if set is None:
            self._set = {0, 1}
        else:
            self._set = set(set)
        super().__init__(*args, **kwargs)

    def _make_gz(self):
        self._gz = [{'g': 'finite_set',
                     'args': {'S': self._set},
                     'range': (0, self.z_size)}]

class Boolean(FiniteSet):
    def __init__(self, *args, **kwargs):
        _set = {0, 1}
        super().__init__(set=_set, *args, **kwargs)