#!/usr/bin/env python

from simple_rl.tasks import FourRoomMDP
from simple_rl.agents import QLearnerAgent
from simple_rl.run_experiments import run_agents_on_mdp
from eigen_options import EigenOptions
from simple_rl.tasks.grid_world.GridWorldMDPClass import make_grid_world_from_file
import os

def fig6(mdp, eo):
    # Avg number of steps and the number of options used
    ql_agent = QLearnerAgent(name='primitive', alpha=0.1, actions=mdp.get_actions())
    eo_agents = [eo.agent(i) for i in range(1, 350)]

    run_agents_on_mdp(
        [ql_agent] + eo_agents,
        mdp,
        instances=5,
        episodes=100,
        steps=150,
        open_plot=True)
    # TODO: plot using csv


def fig7(mdp, eo):
    # Avg return and the number of episodes
    ql_agent = QLearnerAgent(name='primitive', alpha=0.1, actions=mdp.get_actions())
    eo_agents = [eo.agent(i) for i in (2, 4, 8, 64, 128, 256)]

    run_agents_on_mdp(
        [ql_agent] + eo_agents,
        mdp,
        instances=5,
        episodes=500,
        steps=150,
        open_plot=True)
    # TODO: plot using csv

if __name__ == "__main__":
    mdp = make_grid_world_from_file("/Users/noah/Desktop/csci2651F/simple_rl/simple_rl/tasks/grid_world/txt_grids/four_rooms_grid.txt", num_goals=1, randomize=True)
    mdp.gamma = 0.9
    eo = EigenOptions(mdp)

    # fig6(mdp, eo)
    fig7(mdp, eo)
