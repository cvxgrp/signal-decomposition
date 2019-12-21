# -*- coding: utf-8 -*-
''' Component abstract base class module

This module contains the abstract base class for components

Author: Bennet Meyers
'''

from abc import ABC, abstractmethod
import cvxpy as cvx


class Component(ABC):
    def __init__(self, **kwargs):
        self.__parameters = self._get_params()
        self.__cost = self._get_cost()
        for key in ['vmin', 'vmax', 'vavg', 'period']:
            if key in kwargs.keys():
                setattr(self, '_' + key, kwargs[key])
                del kwargs[key]
            else:
                setattr(self, '_' + key, None)
        self.set_parameters(**kwargs)
        return

    @property
    def cost(self):
        if self.__cost is None:
            cost = self._get_cost()
            self.__cost = cost
        else:
            cost = self.__cost
        return cost

    @property
    @abstractmethod
    def is_convex(self):
        return NotImplementedError

    @property
    def cost(self):
        return self.__cost

    @abstractmethod
    def _get_cost(self):
        return NotImplementedError

    @property
    def vmin(self):
        return self._vmin

    @property
    def vmax(self):
        return self._vmax

    @property
    def vavg(self):
        return self._vavg

    @property
    def period(self):
        return self._period

    def make_constraints(self, x):
        c = []
        if self.vmin is not None:
            c.append(x >= self.vmin)
        if self.vmax is not None:
            c.append(x <= self.vmax)
        if self.vavg is not None:
            n = x.size
            c.append(cvx.sum(x) / n == self.vavg)
        if self.period is not None:
            pass
        return c

    @property
    def parameters(self):
        return self.__parameters

    def _get_params(self):
        return None

    def set_parameters(self, **kwargs):
        if self.__parameters is not None:
            for key, value in kwargs.items():
                self.__parameters[key].value = value
        return