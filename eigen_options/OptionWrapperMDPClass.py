from copy import deepcopy

TERMINATE = "AHHHHHHHH"

class OptionWrapperMDP(MDP):
    def __init__(self, MDPClass, eigenvector, state2id):
        self.base_mdp = deepcopy(MDPClass)
        self.state2id = state2id

        self._reward_func = get_eigen_purpose(eigenvector)
        self.eigen_purpose = eigen_purpose

        self.actions = self.base_mdp.actions + [TERMINATE]
        MDP.__init__(self, self.actions, self._transition_func, self._reward_func, init_state=self.base_mdp.init_state, gamma=self.base_mdp.gamma)

    def get_eigen_purpose(eigenvector):
        def reward_function(self, state, action):
            if action == TERMINATE:
                return 0
            else:
                next_state = self.transition_func(state, action)
                return eigenvector[self.state2id(next_state)] - eigenvector[self.state2id(state)]
            
        return reward_function

    def _transition_func(self, state, action):
        if action == TERMINATE:
            return state
        else:
            return self.base_mdp._transition_func(state, action)
            
    

