#!/usr/bin/env python

from simple_rl.agents import QLearnerAgent
from simple_rl.abstraction import AbstractionWrapper, ActionAbstraction
import numpy as np
from scipy.sparse import csgraph

class EigenOptions():
    def __init__(self, mdp):
        self.mdp = mdp

        self._estimate_adjacency_matrix()
        self._calculate_eigen_options()

    def agent(self, num_options):
        eigen_options_aa = ActionAbstraction(self.eigen_options[:num_options])
        return AbstractionWrapper(
            QLearnerAgent,
            actions=self.mdp.get_actions(),
            action_abstr=eigen_options_aa,
            name_ext=str(num_options) + ' options')

    def _estimate_adjacency_matrix(self):
        # TODO: estimate G
        self.G = np.arange(5) * np.arange(5)[:, np.newaxis]  # placeholder

    def _calculate_eigen_options(self):
        # Normalized Graph Laplacian
        L = csgraph.laplacian(self.G)
        w, v = np.linalg.eig(L)
        eigens = sorted(zip(w, v)) # the smallest eigen values first

        # TODO: calculate eigen_options using eigen values and eigen vectors
        self.eigen_options = []

