import unittest
from pathlib import Path
import numpy as np
from osd import Problem
from osd.classes import (
    MeanSquareSmall,
    TimeSmoothEntryClose,
    LinearTrend
)
from osd.classes.wrappers import make_columns_equal

rms = lambda x: np.sqrt(np.average(np.power(x, 2)))

VERBOSE = False

class TestVectorProblem(unittest.TestCase):
    def test_cvx(self):
        y, X_real = make_data()
        classes = [
            MeanSquareSmall(size=y.size),
            TimeSmoothEntryClose(
                lambda1=1e3, lambda2=1e-2, weight=5e-3 / y.size
            ),
            make_columns_equal(LinearTrend)(first_val=0),
        ]
        problem = Problem(y, classes=classes)
        problem.decompose(how='cvx', verbose=VERBOSE)
        np.testing.assert_(np.isclose(problem.problem.value,
                                      problem.objective_value))
        np.testing.assert_(
            problem.objective_value <= 0.0284,
            'actual value: {:.3e}'.format(problem.objective_value)
        )
        # np.testing.assert_(
        #     rms(problem.estimates[0] - X_real[0]) <= 0.11,
        #     'actual value: {:.2e}'.format(rms(problem.estimates[0] - X_real[0]))
        # )
        # np.testing.assert_(
        #     rms(problem.estimates[1] - X_real[1]) <= 0.22,
        #     'actual value: {:.2e}'.format(rms(problem.estimates[1] - X_real[1]))
        # )
        # np.testing.assert_(
        #     rms(problem.estimates[2] - X_real[2]) <= 0.16,
        #     'actual value: {:.2e}'.format(rms(problem.estimates[2] - X_real[2]))
        # )

    def test_admm(self):
        y, X_real = make_data()
        classes = [
            MeanSquareSmall(size=y.size),
            TimeSmoothEntryClose(
                lambda1=1e3, lambda2=1e-2, weight=5e-3 / y.size
            ),
            make_columns_equal(LinearTrend)(first_val=0),
        ]
        problem = Problem(y, classes=classes)
        problem.decompose(how='admm', verbose=VERBOSE)
        np.testing.assert_(
            problem.objective_value <= 0.0284,
            'actual value: {:.3e}'.format(problem.objective_value)
        )
        # np.testing.assert_(
        #     rms(problem.estimates[0] - X_real[0]) <= 0.11,
        #     'actual value: {:.2e}'.format(
        #         rms(problem.estimates[0] - X_real[0]))
        # )
        # np.testing.assert_(
        #     rms(problem.estimates[1] - X_real[1]) <= 0.62,
        #     'actual value: {:.2e}'.format(
        #         rms(problem.estimates[1] - X_real[1]))
        # )
        # np.testing.assert_(
        #     rms(problem.estimates[2] - X_real[2]) <= 0.61,
        #     'actual value: {:.2e}'.format(
        #         rms(problem.estimates[2] - X_real[2]))
        # )

    def test_bcd(self):
        y, X_real = make_data()
        classes = [
            MeanSquareSmall(size=y.size),
            TimeSmoothEntryClose(
                lambda1=1e3, lambda2=1e-2, weight=5e-3 / y.size
            ),
            make_columns_equal(LinearTrend)(first_val=0),
        ]
        problem = Problem(y, classes=classes)
        problem.decompose(how='bcd', verbose=VERBOSE)
        np.testing.assert_(
            problem.objective_value <= 0.0284,
            'actual value: {:.3e}'.format(problem.objective_value)
        )
        # np.testing.assert_(
        #     rms(problem.estimates[0] - X_real[0]) <= 0.11,
        #     'actual value: {:.2e}'.format(
        #         rms(problem.estimates[0] - X_real[0]))
        # )
        # np.testing.assert_(
        #     rms(problem.estimates[1] - X_real[1]) <= 0.25,
        #     'actual value: {:.2e}'.format(
        #         rms(problem.estimates[1] - X_real[1]))
        # )
        # np.testing.assert_(
        #     rms(problem.estimates[2] - X_real[2]) <= 0.20,
        #     'actual value: {:.2e}'.format(
        #         rms(problem.estimates[2] - X_real[2]))
        # )


def make_data():
    filepath = Path(__file__).parent.parent
    data_file_path = (
            filepath
            / "fixtures"
            / "vector_smooth_low_var.txt"
    )
    with open(data_file_path) as file:
        data = np.loadtxt(file)
    T, p = data.shape
    np.random.seed(110100100)
    X1 = 0.15 * np.random.randn(T, p)
    X2 = data
    X3 = np.tile(np.linspace(0, 2, T), (p, 1)).T
    X_real = np.array([X1, X2, X3])
    y = np.sum(X_real, axis=0)
    return y, X_real