# -*- coding: utf-8 -*-
''' Utilities Module

This module contains utility functions for the osd package


Author: Bennet Meyers
'''

import functools
import sys
from time import time
import scipy.sparse as sp

import numpy as np


def compose(*functions):
    """
    Create a create a composition of two or more functions.
    More information: https://mathieularose.com/function-composition-in-python/

    :param functions: Functions to be composed, e.g. f(x), g(x), and h(x)
    :return: Composed function, e.g. f(g(h(x)))
    """
    def compose2(f, g):
        return lambda x: f(g(x))
    return functools.reduce(compose2, functions, lambda x: x)

class AlgProgress():
    def __init__(self, total, start_time):
        self.total = total
        self.ti = start_time
        self.count = 1

    def print(self, obj_val, residual, stop_tol, done=False):
        td = time() - self.ti
        if done:
            self.count -= 1
        if td < 60:
            info = (self.count, td, obj_val, residual, stop_tol)
            msg = '{} iterations, {:.2f} sec -- obj_val: {:.2e}, r: {:.2e},'
            msg += ' tol: {:.2e}      '
        else:
            info = (self.count, td / 60, obj_val, residual, stop_tol)
            msg = '{} iterations, {:.2f} min -- obj_val: {:.2e}, r: {:.2e},'
            msg += ' tol: {:.2e}      '
        if not done:
            progress(self.count, self.total, msg.format(*info), bar_length=20,
                     show_percents=False)
        else:
            progress(self.total, self.total, msg.format(*info), bar_length=20,
                     show_percents=False)
            print('\n')
        self.count += 1
        return


def progress(count, total, status='', bar_length=60, show_percents=True):
    """
    Python command line progress bar in less than 10 lines of code. · GitHub
    https://gist.github.com/vladignatyev/06860ec2040cb497f0f3
    :param count: the current count, int
    :param total: to total count, int
    :param status: a message to display
    :return:
    """
    bar_len = bar_length
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    if show_percents:
        sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    else:
        sys.stdout.write('[%s] ...%s\r' % (bar, status))
    sys.stdout.flush()


def make_estimate(y, X, use_ix, residual_term=0):
    """
    After any given iteration of the ADMM algorithm, generate an estimate that
    is feasible with respect to the global equality constraint by making x0
    equal to the residual between the input data y and the rest of the
    component estimates

    :param y: numpy array containing problem data
    :param X: current estimate of decomposed signal classes from ADMM
    :param use_ix: the known index set (Boolean array)
    :return: the estimate with the first component replaced by the residuals
    """
    X_tilde = np.copy(X)
    sum_ix = np.arange(X.shape[0])
    sum_ix = np.delete(sum_ix, residual_term)
    X_tilde[residual_term, use_ix] = y[use_ix] - np.sum(X[sum_ix][:, use_ix],
                                                        axis=0)
    X_tilde[residual_term, ~use_ix] = 0
    return X_tilde


def calc_obj(y, X, components, use_ix, residual_term=0):
    """
    Calculate the current objective value of the problem

    :param y: numpy array containing problem data
    :param X: current estimate of decomposed signal classes from ADMM
    :param use_ix: the known index set (Boolean array)
    :return: the scalar problem objective value
    """
    if use_ix is None:
        use_ix = ~np.isnan(y)
    K = len(components)
    X_tilde = make_estimate(y, X, use_ix, residual_term=residual_term)
    obj_val = 0
    for k in range(K):
        try:
            cost = components[k].cost(X_tilde[k]).value.item()
        except AttributeError:
            cost = components[k].cost(X_tilde[k])
        weight = components[k].weight
        obj_val += weight * cost
    return obj_val

