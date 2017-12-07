"""
run.py

Core file for discovering options, and running Q-Learning to plot figures from "A Laplacian Framework
for Option Discovery in Reinforcement Learning."
"""
from discovery.eigen_options import EigenOptions
from environment.gridworld import GridWorld
from learning.q_learning import QLearner
import numpy as np

# Parameters for Figure 7
PATH_TO_MDP = "4_Rooms.mdp"
ALPHA, GAMMA, EPSILON = 0.1, 0.9, 1.0
MAX_EPISODE_LEN, MAX_NUM_EPISODES = 100, 549
NUM_SEEDS, NUM_OPTIONS_TO_EVAL = 25, [2, 4, 8, 64, 128, 256]


def moving_average(d, n=50):
    ret = np.cumsum(d, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


if __name__ == "__main__":
    # Load MDP from File
    gridworld = GridWorld(PATH_TO_MDP)

    # Discover Options
    eo = EigenOptions(gridworld, GAMMA)

    # Q-Learning with Options
    eval_returns, learn_returns = [], []
    for idx, num_options in enumerate(NUM_OPTIONS_TO_EVAL):
        # Setup Buffers to Record Experimental Results
        eval_returns.append([])
        learn_returns.append([])
        action_set = gridworld.get_actions()

        # Add Options to action_set
        for i in range(num_options):
            action_set.append(eo.options[i])

        for s in range(NUM_SEEDS):
            eval_returns[idx].append([])
            learn_returns[idx].append([])

            # Q-Learn!
            q = QLearner(ALPHA, GAMMA, EPSILON, gridworld, seed=s, only_primitives=True, action_set=action_set,
                         action_set_per_option=eo.action_sets_per_option)

            for i in range(MAX_NUM_EPISODES):
                learn_returns[idx][s].append(q.learn_episode(timestep_limit=MAX_EPISODE_LEN))
                eval_returns[idx][s].append(q.evaluate_episode(eps=0.01, timestep_limit=MAX_EPISODE_LEN))

    # Q-Learning with Primitive Actions
    eval_returns_primitive, learn_returns_primitive = [], []
    for s in range(NUM_SEEDS):
        eval_returns_primitive.append([])
        learn_returns_primitive.append([])

        # Q-Learn!
        q = QLearner(ALPHA, GAMMA, EPSILON, gridworld, seed=s)

        for i in range(MAX_NUM_EPISODES):
            learn_returns_primitive[s].append(q.learn_episode(timestep_limit=MAX_EPISODE_LEN))
            eval_returns_primitive[s].append(q.evaluate_episode(eps=0.01, timestep_limit=MAX_EPISODE_LEN))

    # Assemble Means, Standard Deviation
    means = {}
    n_prims = eval_returns_primitive
    mn_prims = np.mean(n_prims, axis=0)
    stdn_prims = np.std(n_prims, axis=0)
    means['primitive'] = (mn_prims, stdn_prims, moving_average(mn_prims), eval_returns_primitive)
    for idx, num_options in enumerate(NUM_OPTIONS_TO_EVAL):
        n_opts = eval_returns[idx]
        mn_opts = np.mean(n_opts, axis=0)
        stdn_opts = np.std(n_opts, axis=0)
        means[str(num_options)] = (mn_opts, stdn_opts, moving_average(mn_opts), eval_returns[idx])

    # Save
    import pickle
    with open('means.pik', 'w') as f:
        pickle.dump(means, f)

    # Plot
    import matplotlib.pyplot as plt
    x_vals = range(1, len(means['primitive'][2]) + 1)

    plt.plot(x_vals, means['primitive'][2])
    plt.plot(x_vals, means['2'][2])
    plt.plot(x_vals, means['4'][2])
    plt.plot(x_vals, means['8'][2])
    plt.plot(x_vals, means['64'][2])
    plt.plot(x_vals, means['128'][2])
    plt.plot(x_vals, means['256'][2])
    plt.legend(['Prim', '2', '4', '8', '64', '128', '256'])
    plt.show()

    import IPython
    IPython.embed()