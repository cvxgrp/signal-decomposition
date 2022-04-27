import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

class Problem():
    def __init__(self, data, components):
        self.data = data
        self.components = components
        self.T = components[0]._T
        self.p = components[0]._p       #TODO: allow for p != 1
        self.K = len(components)
        self.decomposition = None

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
            'q': None,
            'r': None,
            'A': A,
            'b': b,
            'g': g
        }
        return out

    def retrieve_result(self, x_value):
        decomposition = np.asarray(x_value[:self.T * self.p * self.K])
        decomposition = decomposition.reshape((self.K, -1))
        self.decomposition = decomposition

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