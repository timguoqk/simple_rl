from copy import deepcopy

TERMINATE = "AHHHHHHHH"

class OptionWrapper(MDP):
    def __init__(self, MDPClass, r_e):
        self.base_mdp = deepcopy(MDPClass)

        self._reward_func = r_e
        self.eigen_purpose = eigen_purpose

        self.actions = self.base_mdp.actions + [TERMINATE]
        MDP.__init__(self, self.actions, self._transition_func, self.reward_func, init_state=self.base_mdp.init_state, gamma=self.base_mdp.gamma)

    def _transition_func(self, state, action):
        if action != TERMINATE:
            return self.base_mdp._transition_func(state, action)
        else:
            return state
    

