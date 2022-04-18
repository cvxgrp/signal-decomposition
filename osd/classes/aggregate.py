import numpy as np
import scipy.sparse as sp
from osd.classes.component import Component
from osd.classes.base_graph_class import GraphComponent

class Aggregate(Component):

    def __init__(self, component_list, **kwargs):
        self.component_list = component_list
        super().__init__(**kwargs)
        return

    @property
    def is_convex(self):
        is_convex = np.alltrue([
            c.is_convex for c in self.component_list
        ])

    def _get_cost(self):
        weights = [c.weight for c in self.component_list]
        costs = [c.cost for c in self.component_list]
        cost = np.sum([w * c for w, c in zip(weights, costs)])
        return cost

    def prox_op(self, v, weight, rho, use_set=None, prox_weights=None):
        raise NotImplementedError
        return

    def make_graph_form(self, T, p):
        gf = AggregateGraph(
            self.component_list, T, p,
            vmin=self.vmin, vmax=self.vmax,
            period=self.period, first_val=self.first_val
        )
        self._gf = gf
        return gf.make_dict()

class AggregateGraph(GraphComponent):
    def __init__(self, component_list, *args, **kwargs):
        self.component_list = component_list
        _ = [c.make_graph_form for c in self.component_list]
        self._gf_list = [c._gf for c in component_list]
        super().__init__(*args, **kwargs)
        return

    def __set_z_size(self):
        self._z_size = np.sum([c.z_size for c in self._gf_list])

    def __make_P(self):
        self._Pz = sp.block_diag([
            c._Pz for c in self._gf_list
        ])
        self._Px = np.sum([
            c._Px for c in self._gf_list
        ])

    def __make_gz(self):
        self._gz = [c._gz for c in self._gf_list]

    def __make_A(self):
        self._A = sp.bmat([[c._A] for c in self._gf_list])