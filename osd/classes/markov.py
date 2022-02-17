''' Markov Process Signal

This module contains the class for a signal defined as a Markov Process over
discrete values

Author: Bennet Meyers
'''

from scipy import sparse
import numpy as np
from osd.classes.component import Component

class MarkovChain(Component):

    def __init__(self, transition_matrix, states=None, **kwargs):
        """


        :param transition_matrix:
        :param states: list-like
        """
        self.P = np.asarray(transition_matrix)
        self.num_states = self.P.shape[0]
        if states is None:
            self.states = np.arange(self.num_states)
        else:
            self.states = states
        super().__init__(**kwargs)
        return

    @property
    def is_convex(self):
        return False

    def _get_cost(self):
        states = self.states
        P = self.P
        def cost(x):
            boolean_state_matrix = np.zeros((len(states), len(x)), dtype=bool)
            for ix, s in enumerate(states):
                boolean_state_matrix[ix] = np.isclose(x, s)
            if not np.all(np.isclose(np.sum(boolean_state_matrix, axis=0), 1)):
                # x not in feasible set, infinite cost
                return np.inf
            state_indices = np.argmax(boolean_state_matrix, axis=0)
            c = 0
            for ix in range(len(x) - 1):
                start = state_indices[ix]
                end = state_indices[ix + 1]
                c += -np.log(P[start, end])
            return c
        return cost



    def prox_op(self, v, weight, rho, use_set=None, prox_weights=None):
        mu = rho / 2 / weight
        num_states = self.P.shape[0]
        distances = np.zeros((num_states, len(v)))
        for i in self.states:
            distances[i] = np.abs(v - i) ** 2
        if use_set is not None:
            distances[:, use_set] = 0
        costs = np.zeros(num_states)
        trajectories = np.zeros_like(distances)
        for i in range(trajectories.shape[1]):
            ix = -(i + 1)
            if ix == -1:
                costs = mu * distances[:, ix]
                trajectories[:, ix] = np.copy(self.states)
            else:
                # size of traj_mat is (num_states x num_states)
                traj_mat = (np.tile(mu * distances[:, ix], num_states).reshape(
                    (num_states, -1), order='F')
                            + np.tile(costs, num_states).reshape(
                            (num_states, -1)))
                non_zero = self.P != 0
                traj_mat[non_zero] += -np.log(self.P[non_zero])
                traj_mat[~non_zero] = np.inf
                costs = np.min(traj_mat, axis=1)
                min_paths = np.argmin(traj_mat, axis=1)
                last_traj = np.copy(trajectories[:, ix + 1:])
                trajectories[:, ix] = self.states
                for i in range(num_states):
                    trajectories[i, ix + 1:] = last_traj[min_paths[i], :]
        min_ix = np.argmin(costs)
        x = trajectories[min_ix]
        return x