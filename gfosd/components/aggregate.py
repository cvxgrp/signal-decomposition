import numpy as np
import scipy.sparse as sp
from gfosd.components.base_graph_class import GraphComponent

class Aggregate(GraphComponent):
    def __init__(self, component_list, *args, **kwargs):
        self._gf_list = component_list
        weight = 1
        super().__init__(weight=weight, *args, **kwargs)
        return

    def prepare_attributes(self, T, p=1):
        helper_removed = False
        for ix, c in enumerate(self._gf_list):
            c.prepare_attributes(T, p=p)
        self._T = T
        self._p = p
        self._x_size = T * p
        self._set_z_size()
        self._Px = self._gf_list[0]._Px # only first one in list can be nonzero
        self._Pz = sp.block_diag([
            c._Pz for c in self._gf_list
        ])
        self._make_q()
        self._make_r()
        self._gx = self._make_gx()
        self._gz = self._make_gz()
        self._make_A()
        self._make_B()
        self._make_c()

    def _set_z_size(self):
        self._z_size = np.sum([c.z_size for c in self._gf_list])

    def _make_P(self):
        self._Pz = sp.block_diag([
            c._Pz for c in self._gf_list
        ])

    def _make_gx(self):
        gx = []
        for ix, component in enumerate(self._gf_list):
            for d in component._gx:
                if isinstance(d, dict):
                    gx.append(d)
        return gx

    def _make_gz(self):
        # print([c._gz for c in self._gf_list])
        gz = []
        z_lengths = [
            entry.z_size for entry in self._gf_list
        ]
        # print(z_lengths)
        breakpoints = np.cumsum(np.r_[[0], z_lengths])
        # print(breakpoints)
        for ix, component in enumerate(self._gf_list):
            pointer = 0
            for d in component._gz:
                if isinstance(d, dict):
                    z_len = np.diff(d['range'])[0]
                    new_d = d.copy()
                    new_d['range'] = (breakpoints[ix] + pointer,
                                      breakpoints[ix] + z_len + pointer)
                    gz.append(new_d)
                    pointer += z_len
        return gz

    def _make_A(self):
        self._A = sp.bmat([[c._A] for c in self._gf_list])

    def _make_B(self):
        self._B = sp.block_diag([
            c._B for c in self._gf_list
        ])

    def _make_c(self):
        self._c = np.concatenate([c._c for c in self._gf_list])
