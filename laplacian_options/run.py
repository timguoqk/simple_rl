"""
run.py

Core file for discovering options, and running Q-Learning to plot figures from "A Laplacian Framework
for Option Discovery in Reinforcement Learning."
"""
from discovery.eigen_options import EigenOptions
from environment.gridworld import GridWorld
from learning.q_learning import QLearner

# Parameters for Figure 7
PATH_TO_MDP = "4_Rooms.mdp"
ALPHA, GAMMA, EPSILON = 0.1, 0.9, 1.0
MAX_EPISODE_LEN, MAX_NUM_EPISODES = 100, 500
NUM_SEEDS, NUM_OPTIONS_TO_EVAL = 5, [2, 4, 8, 64, 128, 256]


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
        for i in range(num_options/2):                # TODO: Paper divides num_options by 2... bug???
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

    import IPython
    IPython.embed()

