"""
eigen_options.py

Core logic for identifying eigenpurposes and eigenbehaviors => Options.
"""
from learning.policy_iteration import PolicyIteration
import numpy as np


class EigenOptions:
    def __init__(self, env, gamma):
        """
        Initialize an EigenOptions discovery interface with a given environment.

        :param env: GridWorld Environment on which to discover Options.
        """
        self.env, self.gamma = env, gamma

        # Compute Laplacian + Eigendecompose
        self.L = self.compute_normalized_laplacian()
        self.eigenvalues, self.eigenvectors = self.eigendecomposition()

        # Calculate Eigenoptions
        self.options, self.action_sets_per_option = self.calculate_options()

    def compute_normalized_laplacian(self):
        """
        Compute the Normalized Laplacian of the Environment's Transition Matrix

        :return: Normalized Laplacian Matrix
        """
        # Get Adjacency Matrix
        adj, num_states = self.env.adj, self.env.num_states
        val = np.zeros([num_states, num_states])

        # Compute Valency Matrix
        for i in xrange(num_states):
            for j in xrange(num_states):
                val[i][i] = np.sum(adj[i])

        # Make Sure Final Matrix will be Full Rank
        for i in xrange(num_states):
            if val[i][i] == 0.0:
                val[i][i] = 1.0

        # Normalized Laplacian
        laplacian = val - adj
        exp_d = self.exponentiate(val, -0.5)
        normalized_laplacian = exp_d.dot(laplacian).dot(exp_d)
        return normalized_laplacian

    def eigendecomposition(self):
        """
        Compute the eigenvalues and eigenvectors of the Normalized Laplacian.
            Note - Uses both directions of the eigenvectors (negates!)

        :return: Tuple of eigenvalues and eigenvectors (including negates)
        """
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eig(self.L)

        # Sort the Eigenvalues and Eigenvectors
        idx = eigenvalues.argsort()[::-1]
        eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]

        # Add Negation (Opposite Direction) of Eigenvectors
        old_eigenvalues, old_eigenvectors = eigenvalues, eigenvectors.T
        eigenvalues, eigenvectors = [], []
        for i in range(len(old_eigenvectors)):
            eigenvalues.append(old_eigenvalues[i])
            eigenvalues.append(old_eigenvalues[i])
            eigenvectors.append(old_eigenvectors[i])
            eigenvectors.append(-1 * old_eigenvectors[i])

        eigenvalues, eigenvectors = np.asarray(eigenvalues), np.asarray(eigenvectors).T
        return eigenvalues, eigenvectors

    def calculate_options(self):
        """
        Calculates the options using the eigenvectors (eigenpurposes) as reward functions.
            - Note: Iterates through Eigenvectors in reverse order

        :return: Tuple of options, action set for given option
        """
        options, action_set_per_option = [], []
        for i in reversed(range(len(self.eigenvectors[0]))):
            print 'Solving for Eigenbehavior %d!' % i

            # Initialize Policy Iteration Learner
            pol_iter = PolicyIteration(self.gamma, self.env, augment_action_set=True)
            self.env.define_reward(self.eigenvectors[:, i])
            v, pi = pol_iter.solve_policy_iteration()

            # Append to Options
            options.append(pi[0:self.env.num_states])
            options_actions = self.env.get_actions()
            options_actions.append('terminate')
            action_set_per_option.append(options_actions)

        # After PVFs, Reset Env Reward
        self.env.define_reward(None)
        self.env.reset()
        return options, action_set_per_option

    @staticmethod
    def exponentiate(m, exp):
        num_rows, num_cols = len(m), len(m[0])
        exp_m = np.zeros([num_rows, num_cols])
        for i in xrange(num_rows):
            for j in xrange(num_cols):
                if m[i][j] != 0:
                    exp_m[i][j] = m[i][j] ** exp
        return exp_m
