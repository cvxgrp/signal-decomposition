''' Quad-Linear utility functions

Author: Bennet Meyers
'''

import scipy.sparse as sp
import numpy as np

def build_constraint_matrix(T, period, vavg, first_val):
    A = None
    if period is not None:
        A = make_periodic_A(T, period)
    if vavg is not None:
        A = sp.bmat([
            [A],
            [np.ones(T)]
        ])
    if first_val is not None:
        row = np.zeros(T)
        row[0] = 1
        A = sp.bmat([
            [A],
            [row]
        ])
    return A

def build_constraint_rhs(T, period, vavg, first_val):
    u = []
    if period is not None:
        u.append(np.zeros(T - period))
    if vavg is not None:
        u.append([vavg * T])
    if first_val is not None:
        u.append([first_val])
    if len(u) > 0:
        u = np.concatenate(u)
    else:
        u = None
    return u

def make_periodic_A(T, period):
    m1 = sp.eye(m=T - period, n=T, k=0)
    m2 = sp.eye(m=T - period, n=T, k=period)
    A = m1 - m2
    return A