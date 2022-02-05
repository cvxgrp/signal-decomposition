# -*- coding: utf-8 -*-
''' Component abstract base class module

This module contains the abstract base class for classes

Author: Bennet Meyers
'''

from abc import ABC, abstractmethod
import numpy as np
import cvxpy as cvx


class Component(ABC):
    def __init__(self, **kwargs):
        self.__parameters = self._get_params()
        self.__cost = self._get_cost()
        for key in ['vmin', 'vmax', 'vavg', 'period', 'first_val', 'weight']:
            if key in kwargs.keys():
                setattr(self, '_' + key, kwargs[key])
                del kwargs[key]
            elif key == 'weight':
                setattr(self, '_' + key, 1)
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

    @abstractmethod
    def prox_op(self, v, weight, rho, use_set=None, prox_weights=None):
        assert not hasattr(super(), 'prox_op')
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

    @property
    def first_val(self):
        return  self._first_val

    @property
    def weight(self):
        return self._weight

    def set_weight(self, weight):
        self._weight = weight

    @property
    def internal_constraints(self):
        if hasattr(self, '_internal_constraints'):
            return self._internal_constraints
        else:
            return None

    def make_constraints(self, x):
        """

        :param x: a cvxpy variable
        :return:
        """
        T = x.shape[0]
        p = 1 if len(x.shape) == 1 else x.shape[1]
        c = []
        if self.vmin is not None:
            c.append(x >= self.vmin)
        if self.vmax is not None:
            c.append(x <= self.vmax)
        if self.vavg is not None:
            if self.period is None:
                n = x.size
                c.append(cvx.sum(x) / n == self.vavg)
            else:
                period = self.period
                c.append(cvx.sum(x[:period]) / period == 0)
        if self.period is not None:
            period = self.period
            c.append(x[:-period] == x[period:])
        if self.first_val is not None:
            c.append(x[0] == self.first_val)
        if self.internal_constraints is not None:
            if isinstance(self.internal_constraints, list):
                for ic in self.internal_constraints:
                    c.append(ic(x, T, p))
            else:
                c.extend(self.internal_constraints(x, T, p))
        return c

    def cvx_prox(self, v, weight, rho, use_set=None, prox_weights=None,
                 **cvx_args):
        """

        :param v:
        :param weight:
        :param rho:
        :param use_set:
        :param prox_weights:
        :return:
        """
        if self.is_convex:
            known_set = ~np.isnan(v)
            if use_set is not None:
                use_set = np.logical_and(known_set, use_set)
            else:
                use_set = known_set
            T = v.shape[0]
            p = 1 if len(v.shape) == 1 else v.shape[1]
            q = np.sum(use_set)
            x = cvx.Variable(v.shape)
            w = weight
            r = rho
            if prox_weights is None:
                cost = w * self.cost(x) + (r / 2) * cvx.sum_squares(
                    x[use_set] - v[use_set])
            else:
                if len(prox_weights) != q:
                    pw = prox_weights[use_set]
                else:
                    pw = prox_weights
                cost = w * self.cost(x) + (r / 2) * cvx.sum_squares(
                    cvx.multiply(pw, x[use_set] - v[use_set]))
            constraints = self.make_constraints(x)
            cvx_prox = cvx.Problem(cvx.Minimize(cost), constraints)
            cvx_prox.solve(**cvx_args)
            return x.value
        else:
            return None

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