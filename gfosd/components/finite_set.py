
import numpy as np
import scipy.sparse as sp
from gfosd.components.base_graph_class import GraphComponent

class FiniteSet(GraphComponent):
    def __init__(self, values=None, *args, **kwargs):
        if values is None:
            self._values = {0, 1}
        else:
            self._values = set(values)
        super().__init__(*args, **kwargs)

    def _make_g(self, size):
        g = [{'g': 'is_finite_set',
                     'args': {'S': self._values},
                     'range': (0, size)}]
        return g

class Boolean(FiniteSet):
    def __init__(self, *args, **kwargs):
        _values = {0, 1}
        super().__init__(values=_values, *args, **kwargs)