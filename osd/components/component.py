# -*- coding: utf-8 -*-
''' Component abstract base class module

This module contains the abstract base class for components

Author: Bennet Meyers
'''

from abc import ABC, abstractmethod

class Component(ABC):
    def __init__(self, *values):
        self.__parameters = self._get_params()
        self.__cost = self._get_cost()
        self.set_parameters(*values)
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
    def constraints(self):
        return []

    @property
    def parameters(self):
        return self.__parameters

    def _get_params(self):
        return None

    def set_parameters(self, *values, ix=None):
        if self.__parameters is not None and ix is None:
            for p, v in zip(self.__parameters, values):
                p.value = v
        elif self.__parameters is not None and ix is not None:
            self.__parameters[ix].value = values[0]
        return