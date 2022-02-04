import unittest
import numpy as np
from scipy.stats import laplace_asymmetric
from osd import Problem
from osd.classes import (
    MeanSquareSmall,
    AsymmetricNoise,
    ConstantChunks,
    LinearTrend
)
from osd.classes.wrappers import make_columns_equal

VERBOSE = False

class TestColsEqual(unittest.TestCase):
    def test_cols_equal_linear(self):
        X1, use_set, T, p = make_X1()
        Xlt = np.tile(np.linspace(-0.5, 1.3, T), (p, 1)).T
        y1 = X1 + Xlt
        y1[~use_set] = np.nan
        c1 = [
            MeanSquareSmall(size=T * p),
            make_columns_equal(LinearTrend)
        ]
        p1 = Problem(y1, c1)
        p1.decompose(how='cvx', reset=True, verbose=VERBOSE)
        np.testing.assert_(
            p1.objective_value <= 0.187,
            'actual value: {:.2e}'.format(p1.objective_value)
        )
        p1.decompose(how='admm', reset=True, verbose=VERBOSE)
        np.testing.assert_(
            p1.objective_value <= 0.224,
            'actual value: {:.2e}'.format(p1.objective_value)
        )
        p1.decompose(how='bcd', reset=True, verbose=VERBOSE)
        np.testing.assert_(
            p1.objective_value <= 0.187,
            'actual value: {:.2e}'.format(p1.objective_value)
        )

    def test_cols_equal_linear_constrained(self):
        X1, use_set, T, p = make_X1()
        Xlt = np.tile(np.linspace(-0.5, 1.3, T), (p, 1)).T
        y1 = X1 + Xlt
        y1[~use_set] = np.nan
        c1 = [
            MeanSquareSmall(size=T * p),
            make_columns_equal(LinearTrend)(first_val=-0.5)
        ]
        p1 = Problem(y1, c1)
        p1.decompose(how='cvx', reset=True, verbose=VERBOSE)
        np.testing.assert_(
            p1.objective_value <= 0.187,
            'actual value: {:.2e}'.format(p1.objective_value)
        )
        p1.decompose(how='admm', reset=True, verbose=VERBOSE)
        np.testing.assert_(
            p1.objective_value <= 0.187,
            'actual value: {:.2e}'.format(p1.objective_value)
        )
        p1.decompose(how='bcd', reset=True, verbose=VERBOSE)
        np.testing.assert_(
            p1.objective_value <= 0.187,
            'actual value: {:.2e}'.format(p1.objective_value)
        )

    def test_cols_equal_asymmetric_noise(self):
        X1, use_set, T, p = make_X1()
        kappa = 2
        np.random.seed(110100100)
        al = laplace_asymmetric.rvs(kappa, size=T)
        Xan = np.tile(al, (p, 1)).T
        y2 = X1 + Xan
        y2[~use_set] = np.nan
        c2 = [
            MeanSquareSmall(size=T * p),
            make_columns_equal(AsymmetricNoise)(weight=1 / (T * p), tau=0.8)
        ]
        p2 = Problem(y2, c2)
        p2.decompose(how='cvx', reset=True, verbose=VERBOSE)
        np.testing.assert_(
            p2.objective_value <= 0.367,
            'actual value: {:.2e}'.format(p2.objective_value)
        )
        p2.decompose(how='admm', reset=True, verbose=VERBOSE)
        np.testing.assert_(
            p2.objective_value <= 0.51,
            'actual value: {:.2e}'.format(p2.objective_value)
        )
        p2.decompose(how='bcd', reset=True, verbose=VERBOSE)
        np.testing.assert_(
            p2.objective_value <= 0.369,
            'actual value: {:.2e}'.format(p2.objective_value)
        )

    def test_cols_equal_constant_chunks(self):
        X1, use_set, T, p = make_X1()
        np.random.seed(110100100)
        cs = 17
        v = np.random.uniform(-1, 1, T // 7 + 1)
        z = np.tile(v, (7, 1))
        z = z.ravel(order='F')
        z = z[:100]
        Xch = np.tile(z, (p, 1)).T
        y3 = X1 + Xch
        y3[~use_set] = np.nan
        c3 = [
            MeanSquareSmall(size=T * p),
            make_columns_equal(ConstantChunks)(length=7)
        ]
        p3 = Problem(y3, c3)
        p3.decompose(how='cvx', reset=True, verbose=VERBOSE)
        np.testing.assert_(
            p3.objective_value <= 0.018,
            'actual value: {:.2e}'.format(p3.objective_value)
        )
        p3.decompose(how='admm', reset=True, verbose=VERBOSE)
        np.testing.assert_(
            p3.objective_value <= 0.024,
            'actual value: {:.2e}'.format(p3.objective_value)
        )
        p3.decompose(how='bcd', reset=True, verbose=VERBOSE)
        np.testing.assert_(
            p3.objective_value <= 0.018,
            'actual value: {:.2e}'.format(p3.objective_value)
        )


def make_X1():
    T = 100
    p = 3
    np.random.seed(110100100)
    X1 = .15 * np.random.randn(T, p)
    use_set = np.random.uniform(size=(T, p)) >= 0.2
    use_set[45:50] = False
    return X1, use_set, T, p