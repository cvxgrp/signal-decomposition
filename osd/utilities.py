# -*- coding: utf-8 -*-
''' Utilities Module

This module contains utility functions for the osd package


Author: Bennet Meyers
'''

import functools

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