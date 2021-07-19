import cvxpy as cvx
import numpy as np
import matplotlib.pyplot as plt
from time import time
from osd.utilities import progress

def count_transitions(seq, num_states):
    C = np.zeros((num_states, num_states))
    e_last = None
    for e in seq:
        if e_last is None:
            e_last = e
            continue
        C[int(e_last), int(e)] += 1
        e_last = e
    return C

def estimate_transition_matrix(seq, num_states):
    counts = count_transitions(seq, num_states)
    state_counts = np.array([np.sum(seq[:-1] == j) for j in range(num_states)])
    m = counts / state_counts.reshape(-1, 1)
    m[np.isnan(m)] = 0
    return m

def plot_mp_run(s, P, P_hat):
    distance = np.linalg.norm((P - P_hat).ravel())
    fig, ax = plt.subplots(ncols=2, figsize=(16, 6))
    ax[0].plot(s)
    ax[0].set_title('Simulated Signal')
    ax[1].stem(P.ravel(), label='true', use_line_collection=True)
    for i in range(P.shape[0] - 1):
        ax[1].axvline(P.shape[0] * i + (P.shape[0] - 0.5), color='red', ls=':', label='_nolegend_')
    ax[1].stem(P_hat.ravel(), markerfmt='C1o', label='estimated', use_line_collection=True)
    ax[1].legend()
    ax[1].set_title('Transition Matrix, distance = {:.2f}'.format(distance))
    return fig

def markov_process_simulator(P, T=500, x_0=None, plot=True):
    P = np.asarray(P)
    s = np.zeros(T)
    n_states = P.shape[0]
    if x_0 is None:
        x_0 = np.random.randint(0, n_states, 1)
    for i in range(T):
        if i == 0:
            s[i] = x_0
        else:
            x_old = s[i-1]
            p = P[int(x_old)]
            choose = np.random.choice(np.arange(n_states), size=1, p=p)
            s[i] = choose
    # Plotting
    if plot:
        # Estimate transition matrix from simulation
        P_hat = estimate_transition_matrix(s, n_states)
        fig = plot_mp_run(s, P, P_hat)
        plt.show()
    return s

def prox1(v, theta, rho):
    r = rho / (2 * theta + rho)
    return r * v

def prox2(v, theta, rho, A=None, return_A=True):
    if A is None:
        n = len(v)
        M = np.diff(np.eye(n), axis=0, n=2)
        r = 2 * theta / rho
        A = np.linalg.inv(np.eye(n) + r * M.T.dot(M))
    if not return_A:
        return A.dot(v)
    else:
        return A.dot(v), A

def prox3_cvx(vec_in, theta_val=1e1, rho_val=0.5, problem=None):
    if problem is None:
        n = len(vec_in)
        theta_over_rho = cvx.Parameter(value=theta_val / rho_val, name='theta_over_rho', pos=True)
        v = cvx.Parameter(n, value=vec_in, name='vec_in')
        x = cvx.Variable(n)
        cost = theta_over_rho * 2 * cvx.norm1(cvx.diff(x)) + cvx.sum_squares(x - v)
        problem = cvx.Problem(cvx.Minimize(cost), [cvx.max(x) <= 1, cvx.min(x) >= -1])
    else:
        parameters = {p.name(): p for p in problem.parameters()}
        parameters['vec_in'].value = vec_in
        if ~np.isclose(theta_val / rho_val, parameters['theta_over_rho'], atol=1e-3):
            parameters['theta'].value = theta_val / rho_val
    problem.solve(solver='MOSEK')
    return x.value, problem

def prox3_noncvx(v, theta, rho, problem=None):
    v1 = np.ones_like(v)
    v2 = np.zeros_like(v)
    d1 = np.abs(v - v1)
    d2 = np.abs(v - v2)
    x = np.ones_like(v1)
    x[d2 < d1] = 0
    return x, None

def calc_obj(y, x2, x3, theta1=1, theta2=1e7, theta3=1e1):
    x1 = y - x2 - x3
    t1 = theta1 * np.sum(np.power(x1, 2))
    t2 = theta2 * np.sum(np.power(np.diff(x2, 2), 2))
    t3 = theta3 * np.sum(np.abs(np.diff(x3, 1)))
    return t1 + t2 + t3

def run_admm(data, num_iter=50, rho=0.5, theta=1e4, verbose=True, switch_at=25, boolean_truth=None, use_ix=None):
    y = data
    A = None
    cvxprob = None
    if use_ix is None:
        use_ix = np.ones_like(data, dtype=bool)
    u = np.zeros_like(y)
    x1 = np.zeros_like(y)
    x2 = np.zeros_like(y)
    x3 = np.zeros_like(y)
#     x1[use_ix] = y[use_ix] / 3
#     x2[use_ix] = y[use_ix] / 3
#     x3[use_ix] = y[use_ix] / 3
    x1[use_ix] = y[use_ix]
#     x2 = problem.estimates[1]
#     x3 = problem.estimates[2]
#     x1 = y - x2 - x3
#     x2 = np.random.uniform(size=len(y))
#     x3 = np.random.uniform(size=len(y))
#     x1 = y - x2 - x3
    residuals = []
    obj_vals = []
    relaxed_obj_vals = []
    boolean_errors = []
    boolean_estimates = np.zeros((num_iter, len(x3)))
    ti = time()
    best = {
        'x1': None,
        'x2': None,
        'x3': None,
        'u': None,
        'it': None,
        'obj_val': np.inf
    }
    for it in range(num_iter):
        if it < switch_at:
            prox3 = prox3_cvx
        else:
            prox3 = prox3_noncvx
        if verbose:
            td = time() - ti
            progress(it, num_iter, '{:.2f} sec'.format(td))
        # Apply proximal operators
        x1 = prox1(x1 - u, 1, rho)
        x2, A = prox2(x2 - u, theta, rho, A=A, return_A=True)
        x3, cxvprob = prox3(x3 - u, 1e0, rho, problem=cvxprob)
        # Consensus step
        u[use_ix] += 2 * (np.average([x1[use_ix], x2[use_ix], x3[use_ix]], axis=0) - y[use_ix] / 3)
        # mean-square-error
        error = np.sum([x1[use_ix], x2[use_ix], x3[use_ix]], axis=0) - y[use_ix]
        mse = np.sum(np.power(error, 2)) / error.size
        residuals.append(mse)
        obj_val = calc_obj(y, x2, x3,theta2=theta, theta3=0)
        obj_vals.append(obj_val)
        obj_val_relax = calc_obj(y, x2, x3, theta2=theta, theta3=1e0)
        relaxed_obj_vals.append(obj_val_relax)
        if boolean_truth is not None:
            boolean_error = np.sum(~np.isclose(np.clip(np.round(x3), 0, 1), boolean_truth))
        else:
            boolean_error = None
        boolean_errors.append(boolean_error)
        boolean_estimates[it] = np.clip(np.round(x3), 0, 1)
        if obj_val < best['obj_val'] and it > switch_at:
                x1_tilde = np.zeros_like(y)
                x1_tilde[use_ix] = y[use_ix] - x2[use_ix] - x3[use_ix]
                best = {
                    'x1': x1_tilde,
                    'x2': x2,
                    'x3': x3,
                    'u': u,
                    'it': it,
                    'obj_val': obj_val
                }
    if verbose:
        td = time() - ti
        progress(it + 1, num_iter, '{:.2f} sec\n'.format(td))
    outdict = {
        'x1': best['x1'],
        'x2': best['x2'],
        'x3': best['x3'],
        'u': best['u'],
        'it': best['it'],
        'residuals': residuals,
        'obj_vals': obj_vals,
        'relaxed_obj_vals': relaxed_obj_vals,
        'boolean_errors': boolean_errors,
        'boolean_estimates': boolean_estimates,
        'best_obj': best['obj_val']
    }
#     outdict = {
#         'x1': y - x2 - x3,
#         'x2': x2,
#         'x3': x3,
#         'u': u,
#         'it': it,
#         'residuals': residuals,
#         'obj_vals': obj_vals
#     }
    return outdict