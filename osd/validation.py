import numpy as np
from sklearn.model_selection import train_test_split

class Validate():
    def __init__(self, problem):
        self.problem = problem

    def holdout_validation(self, holdout=0.2, seed=None, rho=None,
                           rho0_scale=None, how=None, num_iter=1e3,
                           verbose=True, reset=True, X_init=None, u_init=None,
                           stop_early=True, abs_tol=1e-5, rel_tol=1e-5,
                           **cvx_kwargs):
        if seed is not None:
            np.random.seed(seed)
        prob = self.problem
        size = prob.T * prob.p
        if self.problem.p == 1:
            known_ixs = np.arange(size)[prob.known_set]
        else:
            known_ixs = np.arange(size)[prob.known_set.ravel(order='F')]
        train_ixs, test_ixs = train_test_split(
            known_ixs, test_size=holdout, random_state=seed
        )

        hold_set = np.zeros(size, dtype=bool)
        use_set = np.zeros(size, dtype=bool)
        hold_set[test_ixs] = True
        use_set[train_ixs] = True
        if prob.p != 1:
            hold_set = hold_set.reshape((prob.T, prob.p), order='F')
            use_set = use_set.reshape((prob.T, prob.p), order='F')
        prob.decompose(use_set=use_set, rho=rho, rho0_scale=rho0_scale,
                       how=how, num_iter=num_iter, verbose=verbose,
                       reset=reset, X_init=X_init, u_init=u_init,
                       stop_early=stop_early, abs_tol=abs_tol, rel_tol=rel_tol,
                       **cvx_kwargs)
        y_hat = np.sum(prob.components[:, hold_set], axis=0)
        hold_y = prob.data[hold_set]
        residuals = hold_y - y_hat
        holdout_cost = np.average(np.power(residuals, 2))
        return holdout_cost

    def boolean_holdout_validation(self, holdout=0.2, seed=None, rho=None,
                           rho0_scale=None, how=None, num_iter=1e3,
                           verbose=True, reset=True, X_init=None, u_init=None,
                           stop_early=True, abs_tol=1e-5, rel_tol=1e-5,
                           **cvx_kwargs):
        pass
