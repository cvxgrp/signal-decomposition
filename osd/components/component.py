# -*- coding: utf-8 -*-
''' Component abstract base class module

This module contains the abstract base class for components

Author: Bennet Meyers
'''

from abc import ABC, abstractmethod

class Component(ABC):

    @property
    @abstractmethod
    def is_convex(self):
        return NotImplemented

    @property
    @abstractmethod
    def cost(self):
        return NotImplemented

    @property
    @abstractmethod
    def constraints(self):
        return NotImplemented