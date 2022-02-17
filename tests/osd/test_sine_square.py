import unittest
import numpy as np
from scipy import signal
from osd import Problem
from osd.classes import (
    MeanSquareSmall,
    SmoothSecondDifference,
    SparseFirstDiffConvex
)

rms = lambda x: np.sqrt(np.average(np.power(x, 2)))

VERBOSE = False

class TestSineSquare(unittest.TestCase):
    def test_cvx(self):
        y, X_real = make_data()
        T = len(y)
        c1 = MeanSquareSmall(size=T)
        c2 = SmoothSecondDifference(weight=1e3 / T)
        c3 = SparseFirstDiffConvex(weight=2e0 / T, vmax=1, vmin=-1)
        classes = [c1, c2, c3]
        problem1 = Problem(y, classes)
        problem1.decompose(how='cvx', verbose=VERBOSE)
        opt_obj_val = problem1.objective_value
        np.testing.assert_(np.isclose(problem1.problem.value,
                                      opt_obj_val))
        np.testing.assert_(opt_obj_val <= 0.096)
        np.testing.assert_(rms(problem1.components[0] - X_real[0]) <= 0.1)
        np.testing.assert_(rms(problem1.components[1] - X_real[1]) <= 0.21)
        np.testing.assert_(rms(problem1.components[2] - X_real[2]) <= 0.25)

    def test_admm(self):
        y, X_real = make_data()
        T = len(y)
        c1 = MeanSquareSmall(size=T)
        c2 = SmoothSecondDifference(weight=1e3 / T)
        c3 = SparseFirstDiffConvex(weight=2e0 / T, vmax=1, vmin=-1)
        classes = [c1, c2, c3]
        problem1 = Problem(y, classes)
        problem1.decompose(how='admm', verbose=VERBOSE)
        opt_obj_val = problem1.objective_value
        np.testing.assert_(opt_obj_val <= 0.096)
        np.testing.assert_(rms(problem1.components[0] - X_real[0]) <= 0.1)
        np.testing.assert_(rms(problem1.components[1] - X_real[1]) <= 0.21)
        np.testing.assert_(
            rms(problem1.components[2] - X_real[2]) <= 0.252,
            'actual value: {:.3e}'.format(
                rms(problem1.components[2] - X_real[2]))
        )

    def test_bcd(self):
        y, X_real = make_data()
        T = len(y)
        c1 = MeanSquareSmall(size=T)
        c2 = SmoothSecondDifference(weight=1e3 / T)
        c3 = SparseFirstDiffConvex(weight=2e0 / T, vmax=1, vmin=-1)
        classes = [c1, c2, c3]
        problem1 = Problem(y, classes)
        problem1.decompose(how='bcd', verbose=VERBOSE)
        opt_obj_val = problem1.objective_value
        np.testing.assert_(opt_obj_val <= 0.096)
        np.testing.assert_(rms(problem1.components[0] - X_real[0]) <= 0.1)
        np.testing.assert_(rms(problem1.components[1] - X_real[1]) <= 0.23)
        np.testing.assert_(rms(problem1.components[2] - X_real[2]) <= 0.27)

class TestSineSquareMasked(unittest.TestCase):
    def test_cvx(self):
        y, X_real = make_masked_data()
        T = len(y)
        c1 = MeanSquareSmall(size=T)
        c2 = SmoothSecondDifference(weight=1e3 / T)
        c3 = SparseFirstDiffConvex(weight=2e0 / T, vmax=1, vmin=-1)
        classes = [c1, c2, c3]
        problem1 = Problem(y, classes)
        problem1.decompose(how='cvx', verbose=VERBOSE)
        opt_obj_val = problem1.objective_value
        np.testing.assert_(opt_obj_val <= 0.081)
        rms1 = rms(problem1.components[0, problem1.use_set] -
                   X_real[0, problem1.use_set])
        rms2 =rms(problem1.components[1, problem1.use_set] -
                  X_real[1, problem1.use_set])
        rms3 = rms(problem1.components[2, problem1.use_set] -
                   X_real[2, problem1.use_set])
        np.testing.assert_(rms1 <= 0.14)
        np.testing.assert_(rms2 <= 1.04)
        np.testing.assert_(rms3 <= 0.81)

    def test_admm(self):
        y, X_real = make_masked_data()
        T = len(y)
        c1 = MeanSquareSmall(size=T)
        c2 = SmoothSecondDifference(weight=1e3 / T)
        c3 = SparseFirstDiffConvex(weight=2e0 / T, vmax=1, vmin=-1)
        classes = [c1, c2, c3]
        problem1 = Problem(y, classes)
        problem1.decompose(how='admm', verbose=VERBOSE)
        opt_obj_val = problem1.objective_value
        np.testing.assert_(opt_obj_val <= 0.081)
        rms1 = rms(problem1.components[0, problem1.use_set] -
                   X_real[0, problem1.use_set])
        rms2 = rms(problem1.components[1, problem1.use_set] -
                   X_real[1, problem1.use_set])
        rms3 = rms(problem1.components[2, problem1.use_set] -
                   X_real[2, problem1.use_set])
        np.testing.assert_(rms1 <= 0.14)
        np.testing.assert_(rms2 <= 1.04)
        np.testing.assert_(rms3 <= 0.81)

    def test_bcd(self):
        y, X_real = make_masked_data()
        T = len(y)
        c1 = MeanSquareSmall(size=T)
        c2 = SmoothSecondDifference(weight=1e3 / T)
        c3 = SparseFirstDiffConvex(weight=2e0 / T, vmax=1, vmin=-1)
        classes = [c1, c2, c3]
        problem1 = Problem(y, classes)
        problem1.decompose(how='bcd', verbose=VERBOSE)
        opt_obj_val = problem1.objective_value
        np.testing.assert_(opt_obj_val <= 0.081)
        rms1 = rms(problem1.components[0, problem1.use_set] -
                   X_real[0, problem1.use_set])
        rms2 = rms(problem1.components[1, problem1.use_set] -
                   X_real[1, problem1.use_set])
        rms3 = rms(problem1.components[2, problem1.use_set] -
                   X_real[2, problem1.use_set])
        np.testing.assert_(rms1 <= 0.14)
        np.testing.assert_(rms2 <= 1.04)
        np.testing.assert_(rms3 <= 0.81)

def make_masked_data():
    y, X_real = make_data()
    y[55:90] = np.nan
    y[270:360] = np.nan
    X_real[:, np.isnan(y)] = np.nan
    return y, X_real

def make_data():
    """
    a sine wave plus a square wave at a different frequency, and Gaussian noise
    """
    np.random.seed(42)
    t = np.linspace(0, 1000, 200)
    signal1 = np.sin(2 * np.pi * t * 1 / (500.))
    signal2 = signal.square(2 * np.pi * t * 1 / (450.))
    X_real = np.zeros((3, len(t)), dtype=float)
    X_real[0] = 0.15 * np.random.randn(len(signal1))
    X_real[1] = signal1
    X_real[2] = signal2
    y = np.sum(X_real, axis=0)
    return y, X_real