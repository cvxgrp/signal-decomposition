import scipy.sparse as sp
from gfosd.components.base_graph_class import GraphComponent

class SumSquare(GraphComponent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        return

    def _make_P(self):
        self._Pz = (self.weight / self.z_size) * sp.eye(self.z_size)

class SumAbs(GraphComponent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        return

    def _make_gz(self):
        self._gz = [{'g': 'abs',
                     'args': {'weight': self.weight},
                     'range': (0, self.z_size)}]

class SumHuber(GraphComponent):
    def __init__(self, M=1, *args, **kwargs):
        self._M = M
        super().__init__(*args, **kwargs)
        return

    def _make_gz(self):
        self._gz = [{'g': 'huber',
                     'args': {'weight': self.weight, 'M': self._M},
                     'range': (0, self.z_size)}]

class SumQuantile(GraphComponent):
    def __init__(self, tau, *args, **kwargs):
        self.tau = tau
        super().__init__(*args, **kwargs)
        return

    def _make_gz(self):
        self._gz = [{'g': 'quantile',
                     'args': {'weight': self.weight, 'tau': self.tau},
                     'range': (0, self.z_size)}]

class SumCard(GraphComponent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        return

    def _make_gz(self):
        self._gz = [{'f': 'card',
                     'args': {'weight': self.weight},
                     'range': (0, self.z_size)}]