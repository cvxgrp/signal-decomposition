# -*- coding: utf-8 -*-
''' Utilities Module

This module contains utility functions for the osd package


Author: Bennet Meyers
'''

import functools
import sys

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


def progress(count, total, status='', bar_length=60):
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

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()