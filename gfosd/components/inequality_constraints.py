
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

    def _make_g(self, size):
        if self._kind == 'box':
            g = [{'g': 'is_bound',
                  'args': {'lb': self._min,
                           'ub': self._max},
                  'range': (0, size)}]
        elif self._kind == 'max':
            g = [{'g': 'is_neg',
                  'args': {'shift': self._max},
                  'range': (0, size)}]
        elif self._kind == 'min':
            g = [{'g': 'is_pos',
                  'args': {'shift': self._min},
                  'range': (0, size)}]
        return g


