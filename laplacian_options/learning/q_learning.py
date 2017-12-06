"""
q_learning.py

Implementation of Q-Learning with Options!
"""
import numpy as np
import random


class QLearner:
    def __init__(self, alpha, gamma, epsilon, environment, seed=1, only_primitives=False, action_set=None,
                 action_set_per_option=None):
        """
        Initialize a Q-Learner with the necessary hyperparameters.

        :param alpha:
        :param gamma:
        :param epsilon:
        :param environment:
        :param seed:
        :param only_primitives:
        :param action_set:
        :param action_set_per_option:
        """
        self.env, self.gamma, self.alpha, self.epsilon = environment, gamma, alpha, epsilon
        self.num_states, self.num_primitives = self.env.num_states, len(self.env.get_actions())
        self.only_primitives = only_primitives
        random.seed(seed)

        if action_set is None:
            self.action_set = self.env.get_actions()
        else:
            self.action_set, self.options_action_set = action_set, action_set_per_option

        if self.only_primitives:
            if self.epsilon != 1.0:
                print 'Something will go wrong. Epsilon should be 1.0 when using the options only for exploration.'
            self.Q = np.zeros([self.num_states, self.num_primitives])
        else:
            self.Q = np.zeros([self.num_states, len(self.action_set)])

    def get_available_actions(self, s):
        available = []
        for i in range(len(self.action_set)):
            if i < self.num_primitives:
                available.append(i)
            elif self.get_primitive(s, i) != 'terminate':
                available.append(i)
        return available

    def get_id_from_primitive(self, action):
        for i in xrange(self.num_primitives):
            if self.env.get_actions()[i] == action:
                return i
        return 'error'

    def epsilon_greedy(self, f, s, epsilon=None):
        """Epsilon-greedy function. f needs to be Q[s], so it consists of one value per action"""
        rnd = random.uniform(0, 1)
        if epsilon is None:
            epsilon = self.epsilon
        available = self.get_available_actions(s)
        if rnd <= epsilon:
            idx = random.randrange(0, len(available))
            return available[idx]
        else:
            if self.only_primitives:
                available = range(len(self.env.get_actions()))
            t = f[available]
            idx = np.random.choice(np.where(t == t.max())[0])
            return available[idx]

    def get_primitive(self, s, a):
        if a < self.num_primitives:
            return self.action_set[a]
        else:
            idxOption = a - self.num_primitives
            return self.options_action_set[idxOption][self.action_set[a][s]]

    def learn_episode(self, timestep_limit=1000):
        """Execute Q-learning for one episode"""
        self.env.reset()
        r, timestep, previous, cumulative_r, s = 0, 0, -1, 0, self.env.get_curr()

        while not self.env.is_terminal() and timestep < timestep_limit:
            if previous < self.num_primitives:
                a = self.epsilon_greedy(self.Q[s], s)
            action = self.get_primitive(s, a)

            if action == 'terminate':
                a = self.epsilon_greedy(self.Q[s], s)
                action = self.get_primitive(s, a)

            previous, r = a, self.env.act(action)
            cumulative_r += r
            s_next = self.env.get_curr()

            if self.only_primitives:
                a = self.get_id_from_primitive(action)

            self.Q[s][a] = self.Q[s][a] + self.alpha * (r + self.gamma * np.max(self.Q[s_next]) - self.Q[s][a])
            s = s_next
            timestep += 1
        return cumulative_r

    def evaluate_episode(self, eps=None, timestep_limit=1000):
        """Evaluate Q-learning for one episode."""
        self.env.reset()
        r, timestep, previous, cumulative_r, s = 0, 0, -1, 0, self.env.get_curr()

        while not self.env.is_terminal() and timestep < timestep_limit:
            if previous < self.num_primitives:
                a = self.epsilon_greedy(self.Q[s], s, epsilon=eps)

            action = self.get_primitive(s, a)

            if action == 'terminate':
                a = self.epsilon_greedy(self.Q[s], s, epsilon=eps)
                action = self.get_primitive(s, a)

            previous, r = a, self.env.act(action)
            cumulative_r += r
            s_next = self.env.get_curr()
            s = s_next
            timestep += 1
        return cumulative_r
