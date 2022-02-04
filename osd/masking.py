''' Masking Module

This module contains utility functions working with known set masks


Author: Bennet Meyers
'''

import scipy.sparse as sp
import numpy as np

class Mask():
    def __init__(self, use_set):
        if len(use_set.shape) == 1:
            p = 1
            T = len(use_set)
        else:
            T, p, = use_set.shape
        self.use_set = use_set
        self.T = T
        self.p = p
        self.q = np.sum(use_set)
        self.M = make_mask_matrix(use_set)
        self.M_star = make_inverse_mask_matrix(use_set)
        self.MstM = make_masked_identity_matrix(use_set)

    def mask(self, v):
        if self.p == 1:
            vi = v
        else:
            vi = v.ravel(order='F')
        out = self.M @ vi
        return out

    def unmask(self, v):
        out = self.M_star @ v
        if self.p != 1:
            T, p = self.T, self.p
            out = out.reshape((T, p), order='F')
        return out

    def zero_fill(self, v):
        if self.p == 1:
            vi = v
        else:
            vi = v.ravel(order='F')
        out = self.MstM @ vi
        if self.p != 1:
            T, p = self.T, self.p
            out = out.reshape((T, p), order='F')
        return out


def make_mask_matrix(use_set):
    if len(use_set.shape) == 1:
        us = np.copy(use_set)
    else:
        us = np.ravel(use_set, order='F')
    n = len(us)
    K = np.sum(use_set)
    data = np.ones(K)
    i = np.arange(K)
    j = np.arange(n)
    j = j[us]
    M = sp.coo_matrix((data, (i, j)), shape=(K, n))
    return M

def make_inverse_mask_matrix(use_set):
    if len(use_set.shape) == 1:
        us = np.copy(use_set)
    else:
        us = np.ravel(use_set, order='F')
    n = len(us)
    K = np.sum(use_set)
    data = np.ones(K)
    j = np.arange(K)
    i = np.arange(n)
    i = i[us]
    M = sp.coo_matrix((data, (i, j)), shape=(n, K))
    return M

def make_masked_identity_matrix(use_set):
    if len(use_set.shape) == 1:
        us = np.copy(use_set)
    else:
        us = np.ravel(use_set, order='F')
    n = len(us)
    K = np.sum(use_set)
    data = np.ones(K)
    i = np.arange(n)
    i = i[us]
    M = sp.coo_matrix((data, (i, i)), shape=(n, n))
    return M

def fill_forward(v, use_set):
    if len(v.shape) == 1:
        fill = np.arange(v.shape[0])
    else:
        fill = np.arange(v.shape[0])[:, np.newaxis]
    idx = np.where(use_set, fill, 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    if len(v.shape) == 1:
        out = v[idx]
    else:
        out = v[idx, np.arange(idx.shape[1])]
    return out

def fill_backward(v, use_set):
    v_temp = v[::-1]
    us_temp = use_set[::-1]
    out_temp = fill_forward(v_temp, us_temp)
    out = out_temp[::-1]
    return out
