"""
policy_iteration.py

Implements policy iteration to solve an MDP and extract the optimal policy. These learned policies
will compose an option to be specified.
"""
import math
import numpy as np


class PolicyIteration:
    def __init__(self, gamma, env, augment_action_set=False):
        """
        Initialize a PolicyIteration Learner with the given discount factor and MDP.

        :param gamma: Discount Factor
        :param env: MDP
        :param augment_action_set: Boolean whether or not to augment action set.
        """
        self.gamma, self.env, self.num_states = gamma, env, env.num_states + 1                        # TODO: Why +1?
        self.V, self.pi = np.zeros(self.num_states + 1), np.zeros(self.num_states + 1, dtype=np.int)  # TODO: Why +2?

        if augment_action_set:
            self.action_set = np.append(env.get_actions(), ['terminate'])
        else:
            self.action_set = env.get_actions()

    def eval_policy(self):
        """Policy evaluation step"""
        delta = 0.0
        for s in xrange(self.num_states):
            v = self.V[s]
            next_s, next_r = self.env.get_next_state_reward(s, self.action_set[self.pi[s]])
            self.V[s] = next_r + self.gamma * self.V[next_s]
            delta = max(delta, math.fabs(v - self.V[s]))
        return delta

    def improve_policy(self):
        """Policy improvement step"""
        policy_stable = True
        for s in range(self.num_states):
            old_action, temp_v = self.pi[s], [0.0] * len(self.action_set)

            # Get all Value-Function estimates
            for i in range(len(self.action_set)):
                next_s, next_r = self.env.get_next_state_reward(s, self.action_set[i])
                temp_v[i] = next_r + self.gamma * self.V[next_s]

            # Take Argmax
            self.pi[s] = np.argmax(temp_v)

            # Break ties always choosing to terminate:
            if math.fabs(temp_v[self.pi[s]] - temp_v[(len(self.action_set) - 1)]) < 0.001:
                self.pi[s] = len(self.action_set) - 1

            # Stability Check
            if old_action != self.pi[s]:
                policy_stable = False
        return policy_stable

    def solve_policy_iteration(self, theta=0.001):
        """Implementation of Policy Iteration, as in Sutton and Barto (2016)."""
        policy_stable = False
        while not policy_stable:
            # Policy evaluation
            delta = self.eval_policy()
            while theta < delta:
                delta = self.eval_policy()

            # Policy improvement
            policy_stable = self.improve_policy()
        return self.V, self.pi

    def solve_policy_evaluation(self, pi, theta=0.001):
        """Implementation of Policy Evaluation, as in Sutton and Barto (2016)."""
        self.V, iteration, delta = np.zeros(self.num_states + 1), 1, 1.0
        while delta > theta:
            delta = 0
            for s in xrange(self.num_states - 1):
                v, temp_sum = self.V[s], 0
                for a in range(len(pi[s])):
                    next_s, next_r = self.env.get_next_state_reward(s, self.action_set[a])
                    temp_sum += pi[s][a] * 1.0 * (next_r + self.gamma * self.V[next_s])
                self.V[s], delta = temp_sum, max(delta, math.fabs(v - self.V[s]))
            if iteration % 1000 == 0:
                print 'Iteration:', iteration, '\tDelta:', delta
            iteration += 1
        return self.V

    def solve_bellman_equations(self, pi, full_action_set, options_action_set):
        """This method generates the Bellman equations using the model available in self.env
           and solves the generated set of linear equations."""
        primitives = 4

        # ax = b
        a_equations = np.zeros((self.num_states, self.num_states))
        b_equations = np.zeros(self.num_states)

        '''
        # V[s] = \sum \pi(a|s) \sum p(s',r|s,a) [r + \gamma V[s']]
        # V[s] = \sum \pi(a|s) 1.0 [r + \gamma V[s']] (assuming determinism)
        # - \sum \pi(a|s) r = -V[s] + \sum \pi(a|s) \gamma V[s']
        '''
        for s in xrange(self.num_states - 1):
            a_equations[s][s] = -1
            for a in xrange(len(pi[s])):
                next_s, next_r = -1, -1
                # If Primitive
                if isinstance(full_action_set[a], basestring):
                    next_s, next_r = self.env.get_next_state_reward(s, full_action_set[a])

                # If Option
                else:
                    next_s, next_r = self.env.get_next_state_reward_option(s, full_action_set[a],
                                                                           options_action_set[a - primitives])

                a_equations[s][next_s] += pi[s][a] * self.gamma
                b_equations[s] -= pi[s][a] * next_r

        for i in xrange(self.num_states):
            has_only_zeros = True
            for j in xrange(self.num_states):
                if a_equations[i][j] != 0.0:
                    has_only_zeros = False

            if has_only_zeros:
                a_equations[i][i] = 1
                b_equations[i] = 0

        expectation = np.linalg.solve(a_equations, b_equations)
        return expectation
