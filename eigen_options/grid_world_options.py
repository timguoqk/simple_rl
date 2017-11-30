import random
import sys
import os
import numpy as np

# Other imports.
from simple_rl.mdp.MDPClass import MDP
from simple_rl.tasks.grid_world.GridWorldStateClass import GridWorldState
from simple_rl.tasks.grid_world import GridWorldMDPClass

class GridWorldOptionsMDPClass(GridWorldMDPClass):
    def __init__(self,
                 width=5,
                 height=3,
                 init_loc=(1, 1),
                 rand_init=False,
                 goal_locs=[(5, 3)],
                 walls=[],
                 is_goal_terminal=False,
                 gamma=0.99,
                 init_state=None,
                 slip_prob=0.0,
                 name="gridworld"):

        super.__init__(width=5,
                        height=3,
                        init_loc=(1, 1),
                        rand_init=False,
                        goal_locs=[(5, 3)],
                        walls=[],
                        is_goal_terminal=False,
                        gamma=0.99,
                        init_state=None,
                        slip_prob=0.0,
                        name="gridworld")

        ACTIONS = 