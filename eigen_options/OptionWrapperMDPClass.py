from simple_rl.mdp.MDPClass import MDP

from copy import deepcopy

TERMINATE = "AHHHHHHHH"

class OptionWrapperMDP(MDP):
    def __init__(self, MDPClass, eigenvector, state2id):
        self.base_mdp = deepcopy(MDPClass)
        self.state2id = state2id

        self._reward_func = self.get_eigen_purpose(eigenvector)

        self.actions = self.base_mdp.actions + [TERMINATE]
        MDP.__init__(self, self.actions, self._transition_func, self._reward_func, init_state=self.base_mdp.init_state, gamma=self.base_mdp.gamma)

    def get_eigen_purpose(self, eigenvector):
        def reward_function(state, action):
            if action == TERMINATE:
                return 0.00000001
            else:
                next_state = self.transition_func(state, action)
                return eigenvector[self.state2id(next_state)] - eigenvector[self.state2id(state)]
            
        return reward_function

    def _transition_func(self, state, action):
        if action == TERMINATE:
            return state
        else:
            return self.base_mdp._transition_func(state, action)
            
class OptionMDP(object):
    def __init__(self, base_mdp, options, actions):
        self.base_mdp = base_mdp
        self.actions = actions + options
        self.options = set(options)
    
    def _transition_func(self, state, action):
        if action in self.options:
            # TODO: Not sure about this handling
            state_ = run_policy(state, action, self.base_mdp)
        else:
            state_ = self.base_mdp._transition_func(state, action)
        
        return state_

def run_policy(state, policy, mdp):
    while state != TERMINATE:
        action = policy.get_value(state)
        next_state = mdp._transition_func(state, action)
        state = next_state
    
    return state