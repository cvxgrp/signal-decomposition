import numpy as np
import scipy.sparse as sp
from gfosd.components.base_graph_class import GraphComponent

class Aggregate(GraphComponent):
    def __init__(self, component_list, *args, **kwargs):
        self._gf_list = component_list
        T = component_list[0]._T
        p = component_list[0]._p
        weight = component_list[0]._weight
        super().__init__(T, p=p, weight=weight, *args, **kwargs)
        return

    def _set_z_size(self):
        self._z_size = np.sum([c.z_size for c in self._gf_list])

    def _make_P(self):
        self._Pz = sp.block_diag([
            c._Pz for c in self._gf_list
        ])

    def _make_gz(self):
        # print([c._gz for c in self._gf_list])
        self._gz = []
        z_lengths = [
            entry.z_size for entry in self._gf_list
        ]
        # print(z_lengths)
        breakpoints = np.cumsum(np.r_[[0], z_lengths])
        # print(breakpoints)
        for ix, component in enumerate(self._gf_list):
            pointer = 0
            for d in component._g:
                if isinstance(d, dict):
                    z_len = np.diff(d['range'])[0]
                    new_d = d.copy()
                    new_d['range'] = (breakpoints[ix] + pointer,
                                      breakpoints[ix] + z_len + pointer)
                    self._gz.append(new_d)
                    pointer += z_len

    def _make_A(self):
        self._A = sp.bmat([[c._A] for c in self._gf_list])

    def _make_B(self):
        self._B = sp.block_diag([
            c._B for c in self._gf_list
        ])

    def _make_c(self):
        self._c = np.concatenate([c._c for c in self._gf_list])