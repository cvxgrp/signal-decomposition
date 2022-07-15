import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import qss
import cvxpy as cvx

class Problem():
    def __init__(self, data, components):
        self.data = data
        self.components = components
        if len(data.shape) == 1:
            T = len(data)
            p = 1
        else:
            T, p = data.shape
        self.T = T
        self.p = p
        for c in self.components:
            c.prepare_attributes(T, p)
        self.K = len(components)
        self.decomposition = None
        self.objective_value = None
        self._qss_soln = None
        self._qss_obj = None

    def make_graph_form(self):
        num_x = self.T * self.p * self.K
        dicts = [c.make_dict() for c in self.components]
        Pz = sp.block_diag([d['Pz'] for d in dicts])
        P = sp.block_diag([sp.dok_matrix(2 * (num_x,)), Pz])
        Al = sp.block_diag([d['A'] for d in dicts])
        Ar = sp.block_diag([d['B'] for d in dicts])
        A = sp.bmat([[Al, Ar]])
        last_block_row = [sp.eye(self.T * self.p)] * self.K
        last_block_row.append(sp.dok_matrix((self.T * self.p,
                                             Pz.shape[0])))
        last_block_row = sp.hstack(last_block_row)
        A = sp.vstack([A, last_block_row])
        b = np.concatenate([d['c'] for d in dicts])
        b = np.concatenate([b, self.data])
        g = []
        z_lengths = [
            entry.z_size for entry in self.components
        ]
        # print(z_lengths)
        breakpoints = np.cumsum(np.r_[[num_x], z_lengths])
        # print(breakpoints)
        for ix, component in enumerate(self.components):
            pointer = 0
            for d in component._g:
                if isinstance(d, dict):
                    z_len = np.diff(d['range'])[0]
                    # print(z_len)
                    new_d = d.copy()
                    new_d['range'] = (breakpoints[ix] + pointer,
                                      breakpoints[ix] + z_len + pointer)
                    g.append(new_d)
                    pointer += z_len
        out = {
            'P': P,
            'q': np.zeros(P.shape[0]),  # not currently used
            'r': 0,                     # not currently used
            'A': A,
            'b': b,
            'g': g
        }
        return out

    def decompose(self, solver='qss', data=None, **kwargs):
        if data is None:
            data = self.make_graph_form()
        if solver.lower() == 'qss':
            result = self._solve_qss(data, **kwargs)
        else:
            result = self._solve_cvx(data, solver, **kwargs)
        self.retrieve_result(result)

    def _solve_qss(self, data, **solver_kwargs):
        solver = qss.QSS(data)
        objval, soln = solver.solve(**solver_kwargs)
        self._qss_soln = soln
        self.objective_value = objval
        self._qss_obj = solver
        # print(soln.T @ data['P'] @ soln)
        return soln

    def _solve_cvx(self, data, solver, **solver_kwargs):
        if solver.lower() in ['cvx', 'cvxpy']:
            solver = None
        x = cvx.Variable(data['P'].shape[0])
        cost = 0.5 * cvx.quad_form(x, data['P'])
        constraints = [data['A'] @ x == data['b']]
        for gfunc in data['g']:
            if gfunc['g'] == 'abs':
                cost += cvx.sum(gfunc['args']['weight'] * cvx.abs(
                    x[gfunc['range'][0]:gfunc['range'][1]]))
            elif gfunc['g'] == 'huber':
                try:
                    M = gfunc['args']['M']
                except KeyError:
                    M = 1
                cost += cvx.sum(gfunc['args']['weight'] * cvx.huber(
                    x[gfunc['range'][0]:gfunc['range'][1]], M=M))
            elif gfunc['g'] == 'is_pos':
                constraints.append(x[gfunc['range'][0]:gfunc['range'][1]] >= 0)
            elif gfunc['g'] == 'is_neg':
                constraints.append(x[gfunc['range'][0]:gfunc['range'][1]] <= 0)
            elif gfunc['g'] == 'is_bound':
                constraints.extend([x[gfunc['range'][0]:gfunc['range'][1]] >= 0,
                                    x[gfunc['range'][0]:gfunc['range'][1]] <= 1])
            elif gfunc['g'] in ['card', 'is_finite_set']:
                print('Problem is non-convex and is not solvable with CVXPY.')
                print('Please try QSS.')
                self.objective_value = None
                return
        objective = cvx.Minimize(cost)
        cvx_prob = cvx.Problem(objective, constraints)
        cvx_prob.solve(solver=solver, **solver_kwargs)
        self.objective_value = cvx_prob.value
        return x.value

    def retrieve_result(self, x_value):
        if x_value is not None:
            decomposition = np.asarray(x_value[:self.T * self.p * self.K])
            decomposition = decomposition.reshape((self.K, -1))
            self.decomposition = decomposition
        else:
            self.decomposition = None

    def plot_decomposition(self, x_series=None, X_real=None, figsize=(10, 8),
                           label='estimated', exponentiate=False,
                           skip=None):
        if self.decomposition is None:
            print('No decomposition available.')
            return
        if not exponentiate:
            f = lambda x: x
            base = 'Component $x'
        else:
            f = lambda x: np.exp(x)
            base = 'Component $\\tilde{x}'
        if skip is not None:
            skip = np.atleast_1d(skip)
            nd = len(skip)
        else:
            nd = 0
        K = len(self.decomposition)
        fig, ax = plt.subplots(nrows=K + 1 - nd, sharex=True, figsize=figsize)
        if x_series is None:
            xs = np.arange(self.decomposition.shape[1])
        else:
            xs = np.copy(x_series)
        ax_ix = 0
        for k in range(K + 1):
            if skip is not None and k in skip:
                continue
            if k == 0:
                est = self.decomposition[k]
                ax[ax_ix].plot(xs, f(est), label=label, linewidth=1,
                               ls='none', marker='.', ms=2)
                ax[ax_ix].set_title(base + '^{}$'.format(k + 1))
                if X_real is not None:
                    true = X_real[k]
                    ax[ax_ix].plot(true, label='true', linewidth=1)
            elif k < K:
                est = self.decomposition[k]
                ax[ax_ix].plot(xs, f(est), label=label, linewidth=1)
                ax[ax_ix].set_title(base + '^{}$'.format(k + 1))
                if X_real is not None:
                    true = X_real[k]
                    ax[ax_ix].plot(xs, true, label='true', linewidth=1)
            else:
                if not exponentiate:
                    lbl = 'observed, $y$'
                else:
                    lbl = 'observed, $\\tilde{y}$'
                ax[ax_ix].plot(xs, f(self.data), label=lbl,
                           linewidth=1, color='green')
                ax[ax_ix].plot(xs, f(np.sum(self.decomposition[1:], axis=0)),
                               label='denoised estimate', linewidth=1)
                if X_real is not None:
                    ax[ax_ix].plot(xs, np.sum(X_real[1:], axis=0), label='true',
                               linewidth=1)
                ax[ax_ix].set_title('composed signal')
                ax[ax_ix].legend()
            if X_real is not None:
                ax[ax_ix].legend()
            ax_ix += 1
        plt.tight_layout()
        return fig