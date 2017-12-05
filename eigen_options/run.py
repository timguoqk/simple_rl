#!/usr/bin/env python

from simple_rl.tasks import FourRoomMDP
from simple_rl.agents import QLearnerAgent
from simple_rl.run_experiments import run_agents_on_mdp
from eigen_options import EigenOptions


def fig6(mdp, eo):
    # Avg number of steps and the number of options used
    ql_agent = QLearnerAgent(name='primitive', actions=mdp.get_actions())
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
    ql_agent = QLearnerAgent(name='primitive', actions=mdp.get_actions())
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
    mdp = FourRoomMDP()
    eo = EigenOptions(mdp)

    # fig6(mdp, eo)
    fig7(mdp, eo)
