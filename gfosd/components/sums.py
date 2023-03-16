import scipy.sparse as sp
from gfosd.components.base_graph_class import GraphComponent

class SumSquare(GraphComponent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        return

    def _make_P(self, size):
        return self.weight * 2 * sp.eye(size)  # note the (1/2) in canonical form!

class SumAbs(GraphComponent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        return

    def _make_g(self, size):
        g = [{'g': 'abs',
              'args': {'weight': self.weight},
              'range': (0, size)}]
        return g

class SumHuber(GraphComponent):
    def __init__(self, M=1, *args, **kwargs):
        self._M = M
        super().__init__(*args, **kwargs)
        return

    def _make_g(self, size):
        g = [{'g': 'huber',
              'args': {'weight': self.weight, 'M': self._M},
              'range': (0, size)}]
        return g

class SumQuantile(GraphComponent):
    def __init__(self, tau, *args, **kwargs):
        self.tau = tau
        super().__init__(*args, **kwargs)
        return

    def _make_g(self, size):
        g = [{'g': 'quantile',
              'args': {'weight': self.weight, 'tau': self.tau},
              'range': (0, size)}]
        return g

class SumCard(GraphComponent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        return

    def _make_g(self, size):
        g = [{'g': 'card',
              'args': {'weight': self.weight},
              'range': (0, size)}]
        return g
