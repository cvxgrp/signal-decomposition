''' Generator Functions

These are helper functions for creating numerical examples

Author: Bennet Meyers
'''

import numpy as np
import cvxpy as cvx


def proj_l2_d0(data, theta=1, c=1, solver='MOSEK'):
    """Sum of squares"""
    x = data
    y = cvx.Variable(len(x))
    cost = cvx.sum_squares(x - y)
    objective = cvx.Minimize(cost)
    constraints = [theta * cvx.sum_squares(y) <= c]
    problem = cvx.Problem(objective, constraints)
    problem.solve(solver=solver)
    return y.value

def proj_l1_d0(data, theta=1, c=1, solver='MOSEK'):
    """Sum of squares"""
    x = data
    y = cvx.Variable(len(x))
    cost = cvx.sum_squares(x - y)
    objective = cvx.Minimize(cost)
    constraints = [theta * cvx.sum(cvx.abs(y)) <= c]
    problem = cvx.Problem(objective, constraints)
    problem.solve(solver=solver)
    return y.value

def proj_l1_d1(data, theta=1, c=1, solver='MOSEK'):
    """Sum of absolute value of first difference"""
    x = data
    y = cvx.Variable(len(x))
    cost = cvx.sum_squares(x - y)
    objective = cvx.Minimize(cost)
    constraints = [theta * cvx.sum(cvx.abs(cvx.diff(y, k=1))) <= c]
    problem = cvx.Problem(objective, constraints)
    problem.solve(solver=solver)
    return y.value

def proj_l1_d2(data, theta=1, c=1, solver='MOSEK'):
    """Sum of absolute value of second difference"""
    x = data
    y = cvx.Variable(len(x))
    cost = cvx.sum_squares(x - y)
    objective = cvx.Minimize(cost)
    constraints = [theta * cvx.sum(cvx.abs(cvx.diff(y, k=2))) <= c]
    problem = cvx.Problem(objective, constraints)
    problem.solve(solver=solver)
    return y.value

def proj_l2_d2(data, theta=1, c=1, solver='MOSEK'):
    """Sum of squares of second difference"""
    x = data
    y = cvx.Variable(len(x))
    cost = cvx.sum_squares(x - y)
    objective = cvx.Minimize(cost)
    constraints = [theta * cvx.sum_squares(cvx.diff(y, k=2)) <= c]
    problem = cvx.Problem(objective, constraints)
    problem.solve(solver=solver)
    return y.value

def proj_l2_d1(data, theta=1, c=1, solver='MOSEK'):
    """Sum of squares of first difference"""
    x = data
    y = cvx.Variable(len(x))
    cost = cvx.sum_squares(x - y)
    objective = cvx.Minimize(cost)
    constraints = [theta * cvx.sum_squares(cvx.diff(y, k=1)) <= c]
    problem = cvx.Problem(objective, constraints)
    problem.solve(solver=solver)
    return y.value

def make_pwc_data(length, randomize=True, segments=4, points=None, shifts=None):
    if randomize:
        points = np.random.choice(
            np.arange(int(0.1 * length),
                      int(0.9 * length)),
            segments - 1,
            replace=False
        )
        points.sort()
        points = np.r_[points, [length]]
        shifts = np.random.uniform(0.45, 1.25, size=segments - 1)
        shifts *= np.random.randint(0, 2, size=segments - 1) * 2 - 1
        print(shifts)
    elif points is None:
        points = [0, int(length * 0.2), int(length * 0.55), int(length * 0.85), length]
    elif shifts is None:
        shifts = [0, .5, -0.75, .2]
    cp = np.zeros(length)
    value = 0
    for ix, shft in enumerate(shifts):
        a = points[ix]
        b = points[ix + 1]
        value += shft
        cp[a:b] = value
        print(a, b, shft)
    return cp