#!/usr/bin/env python

from simple_rl.agents import QLearnerAgent
from simple_rl.abstraction import AbstractionWrapper, ActionAbstraction
from simple_rl.abstraction.action_abs.PredicateClass import Predicate
from simple_rl.abstraction.action_abs.OptionClass import Option
from simple_rl.run_experiments import run_single_agent_on_mdp
import numpy as np
from scipy.sparse import csgraph
from OptionWrapperMDPClass import OptionWrapperMDP
from simple_rl.planning.ValueIterationClass import ValueIteration
from simple_rl.tasks.grid_world.GridWorldMDPClass import GridWorldMDP


TERMINATE = "AHHHHHHHH"

class EigenOptions:
    def __init__(self, mdp):
        # Get Statistics about MDP
        self.mdp = mdp

        # Get Height, Width => 1-Indexed [1 -> width, 1 -> height] (inclusive)
        self.width, self.height = self.mdp.width, self.mdp.height
        self.num_states = self.width * self.height

        # Compute Adjacency Matrix (num_states x num_states)
        self._compute_adjacency_matrix()

        self._calculate_eigen_options()

    def agent(self, num_options):
        eigen_options_aa = ActionAbstraction(self.eigen_options[:num_options])
        return AbstractionWrapper(QLearnerAgent, actions=self.mdp.get_actions(), action_abstr=eigen_options_aa,
                                  name_ext=str(num_options) + ' options')

    def _compute_adjacency_matrix(self):
        self.G = np.zeros(shape=[self.num_states, self.num_states], dtype=np.int32)
        for i in range(1, self.height + 1):
            for j in range(1, self.width + 1):
                if not self.mdp.is_wall(i, j):
                    # Check if North Square is Empty
                    if (i - 1) >= 1 and not self.mdp.is_wall(i - 1, j):
                        self.G[self._coord2id(i, j)][self._coord2id(i - 1, j)] = 1

                    # Check if South Square is Empty
                    if (i + 1) <= self.height and not self.mdp.is_wall(i + 1, j):
                        self.G[self._coord2id(i, j)][self._coord2id(i + 1, j)] = 1

                    # Check if West Square is Empty
                    if (j - 1) >= 1 and not self.mdp.is_wall(i, j - 1):
                        self.G[self._coord2id(i, j)][self._coord2id(i, j - 1)] = 1

                    # Check if East Square is Empty
                    if (j + 1) <= self.width and not self.mdp.is_wall(i, j + 1):
                        self.G[self._coord2id(i, j)][self._coord2id(i, j + 1)] = 1

    def _coord2id(self, i, j):
        """ Turns coordinate into ID """
        return (i - 1) * self.width + (j - 1)

    def state2id(self, state):
        val = state.y * self.width + state.x
        return int((state.y - 1) * self.width + (state.x - 1))

    def up_policy(self, state):
        return "up"

    def down_policy(self, state):
        return "down"

    def left_policy(self, state):
        return "left"

    def right_policy(self, state):
        return "right"

    def _calculate_eigen_options(self):
        # Normalized Graph Laplacian
        L = csgraph.laplacian(self.G, normed=True)
        w, v = np.linalg.eig(L)
        print w
        eigens = []

        for vector in v:
            eigens.append(vector)   # the smallest eigen values first
            inv = [x*-1 for x in vector]
            eigens.append(inv)

        eigens = eigens[::-1]
        #eigens = [eigens[0]]

        true_predicate = Predicate(lambda x: True)
        

        up_primitive = Option(true_predicate, true_predicate, self.up_policy)
        down_primitive = Option(true_predicate, true_predicate, self.down_policy)
        left_primitive = Option(true_predicate, true_predicate, self.left_policy)
        right_primitive = Option(true_predicate, true_predicate, self.right_policy)
        options = [up_primitive, down_primitive, left_primitive, right_primitive]
        options = []

        for vector in eigens:
            eigen_option_mdp = OptionWrapperMDP(self.mdp, vector, self.state2id)
            vi = ValueIteration(eigen_option_mdp, delta=0.000000001, max_iterations=1000, sample_rate=10)
            print vi.run_vi()

            #self.mdp.visualize_policy(lambda x: str(int(100*round(vi._compute_max_qval_action_pair(x)[0], 2))))
            #self.mdp.visualize_policy(vi.policy)
            term_predicate = Predicate(lambda x: vi.policy(x) == TERMINATE)
            option = Option(true_predicate, term_predicate, vi.policy)
            options.append(option)
        
        self.eigen_options = options


    def visualize_mdp(self):
        for i in range(1, self.width + 1):
            for j in range(1, self.height + 1):
                symbol = 'w' if self.mdp.is_wall(i, j) else '-'
                if list([i, j]) == list(self.mdp.cur_state.features()):
                    symbol = 'X'
                elif (i, j) in self.mdp.goal_locs:
                    symbol = 'G'
                print symbol,
            print
