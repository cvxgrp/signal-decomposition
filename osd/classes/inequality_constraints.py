
import numpy as np
import scipy.sparse as sp
from osd.classes.component import Component
from osd.classes.base_graph_class import GraphComponent

class MinVal(Component):

    def __init__(self, min_val, **kwargs):
        self.min_val = min_val
        super().__init__(**kwargs)
        return

    @property
    def is_convex(self):
        return True

    def _get_cost(self):
        cost = lambda x: 0 if x >= self.min_val else np.inf
        return cost

    def prox_op(self, v, weight, rho, use_set=None, prox_weights=None):
        raise NotImplementedError
        return

    def make_graph_form(self, T, p):
        gf = MinValGraph(
            self.min_val, self.weight, T, p,
            vmin=self.vmin, vmax=self.vmax,
            period=self.period, first_val=self.first_val
        )
        self._gf = gf
        return gf.make_dict()

class MinValGraph(GraphComponent):
    def __init__(self, min_val, *args, **kwargs):
        self.min_val = min_val
        super().__init__(*args, **kwargs)
        return

    def __set_z_size(self):
        self._z_size = self.x_size

    def __make_gz(self):
        self._gz = [{'f': 4,
                     'args': None,
                     'range': (self.x_size, self.x_size + self.z_size)}]

    def __make_A(self):
        self._A = sp.eye(self.x_size)

    def __make_B(self):
        self._B = -1 * sp.eye(self.z_size)

    def __make_c(self):
        self._c = self.min_val * np.ones(self.x_size)

class MaxVal(Component):

    def __init__(self, max_val, **kwargs):
        self.max_val = max_val
        super().__init__(**kwargs)
        return

    @property
    def is_convex(self):
        return True

    def _get_cost(self):
        cost = lambda x: 0 if x <= self.max_val else np.inf
        return cost

    def prox_op(self, v, weight, rho, use_set=None, prox_weights=None):
        raise NotImplementedError
        return

    def make_graph_form(self, T, p):
        gf = MaxValGraph(
            self.max_val, self.weight, T, p,
            vmin=self.vmin, vmax=self.vmax,
            period=self.period, first_val=self.first_val
        )
        self._gf = gf
        return gf.make_dict()


class MinValGraph(GraphComponent):
    def __init__(self, max_val, *args, **kwargs):
        self.max_val = max_val
        super().__init__(*args, **kwargs)
        return

    def __set_z_size(self):
        self._z_size = self.x_size

    def __make_gz(self):
        self._gz = [{'f': 5,
                     'args': None,
                     'range': (self.x_size, self.x_size + self.z_size)}]

    def __make_A(self):
        self._A = sp.eye(self.x_size)

    def __make_B(self):
        self._B = -1 * sp.eye(self.z_size)

    def __make_c(self):
        self._c = self.max_val * np.ones(self.x_size)

class BoxConstraint(Component):

    def __init__(self, min_val, max_val, **kwargs):
        self.min_val = min_val
        self.max_val = max_val
        super().__init__(**kwargs)
        return

    @property
    def is_convex(self):
        return True

    def _get_cost(self):
        cost = lambda x: (0 if (x >= self.min_val and x <= self.max_val)
                          else np.inf)
        return cost

    def prox_op(self, v, weight, rho, use_set=None, prox_weights=None):
        raise NotImplementedError
        return

    def make_graph_form(self, T, p):
        gf = BoxConstraintGraph(
            self.max_val, self.max_val, self.weight, T, p,
            vmin=self.vmin, vmax=self.vmax,
            period=self.period, first_val=self.first_val
        )
        self._gf = gf
        return gf.make_dict()

class BoxConstraintGraph(GraphComponent):
    def __init__(self, min_val, max_val, *args, **kwargs):
        self.min_val = min_val
        self.max_val = max_val
        super().__init__(*args, **kwargs)
        return

    def __set_z_size(self):
        self._z_size = self.x_size

    def __make_gz(self):
        self._gz = [{'f': 6,
                     'args': None,
                     'range': (self.x_size, self.x_size + self.z_size)}]

    def __make_A(self):
        self._A = sp.eye(self.x_size)

    def __make_B(self):
        self._B = -1 * (self.max_val - self.min_val) * sp.eye(self.z_size)

    def __make_c(self):
        self._c = self.min_val * np.ones(self.x_size)