"""
gridworld.py

Reads, parses, and initializes a GridWorld from file.
"""
import numpy as np


class GridWorld:
    def __init__(self, path, use_negative_rewards=False):
        """
        Initializes a GridWorld from the MDP specified in plaintext at the given path.

        :param path: Path to base GridWorld MDP.
        """
        # Parse and Setup MDP
        self.mdp, self.start, self.goal = self.parse(path)
        self.num_rows, self.num_cols = self.mdp.shape
        self.num_states = self.num_rows * self.num_cols
        self.use_negative_rewards = use_negative_rewards
        self.reward_function = None
        self.curr_x, self.curr_y = self.start[0], self.start[1]

        # Build Transition/Adjacency Matrix
        self.adj = self.build_adjacency_matrix()

    @staticmethod
    def parse(path):
        """
        Parse plaintext MDP into the corresponding GridWorld Matrix, with necessary dimensions.

        :param path: Path to plaintext MDP
        :return: Tuple of Matrix MDP (-1 for Walls, 0 for Available), Start Coordinate, Goal Coordinate
        """
        # Open + Read File Plaintext
        with open(path, 'r') as f:
            rows = map(str.strip, f.readlines())

        # Create MDP Matrix
        num_rows, num_cols = len(rows), len(rows[0])
        start, goal = None, None
        mdp = np.zeros([num_rows, num_cols], dtype=np.int32)

        for i in range(num_rows):
            for j in range(num_cols):
                if rows[i][j] == 'W':
                    mdp[i][j] = -1
                else:
                    mdp[i][j] = 0
                    if rows[i][j] == 'S':
                        start = (i, j)
                    if rows[i][j] == 'G':
                        goal = (i, j)

        return mdp, start, goal

    def build_adjacency_matrix(self):
        G = np.zeros(shape=[self.num_states, self.num_states], dtype=np.int32)
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                if not self.is_wall(i, j):
                    # Check if North Square is Empty
                    if not self.is_wall(i - 1, j):
                        G[self.coord2id(i, j)][self.coord2id(i - 1, j)] = 1

                    # Check if South Square is Empty
                    if not self.is_wall(i + 1, j):
                        G[self.coord2id(i, j)][self.coord2id(i + 1, j)] = 1

                    # Check if West Square is Empty
                    if not self.is_wall(i, j - 1):
                        G[self.coord2id(i, j)][self.coord2id(i, j - 1)] = 1

                    # Check if East Square is Empty
                    if not self.is_wall(i, j + 1):
                        G[self.coord2id(i, j)][self.coord2id(i, j + 1)] = 1
        return G

    def coord2id(self, i, j):
        """Turns coordinate into ID"""
        return i * self.num_cols + j

    def id2coord(self, idx):
        """Turns state identifier into coordinate"""
        y = idx % self.num_cols
        x = (idx - y) / self.num_cols
        return x, y

    def is_terminal(self):
        """Returns whether the agent is in a terminal state (or goal)."""
        if self.curr_x == self.goal[0] and self.curr_y == self.goal[1]:
            return True
        else:
            return False

    def is_wall(self, i, j):
        """Checks if given coordinate is a wall"""
        return self.mdp[i, j] == -1

    def define_reward(self, vector):
        """Set Eigenpurpose as reward function"""
        self.reward_function = vector

    @staticmethod
    def get_actions():
        """At first the four directional actions are the ones available."""
        return ['up', 'right', 'down', 'left']

    def reset(self):
        self.curr_x, self.curr_y = self.start[0], self.start[1]

    def get_curr(self):
        """Returns the unique identifier for the current state the agent is."""
        curr_idx = self.coord2id(self.curr_x, self.curr_y)
        return curr_idx

    def act(self, action):
        """Take an action and update state."""
        if self.reward_function is None and self.is_terminal():
            return 0
        else:
            next_x, next_y = self.get_next_state(action)
            reward = self.get_next_reward(self.curr_x, self.curr_y, action, next_x, next_y)
            self.curr_x, self.curr_y = next_x, next_y
            return reward

    def get_next_state(self, action):
        """
        Returns the next state (x, y) given an action. Does not update state, just predicts.

        :param action: Action to take from current state
        :return: Next State Coordinate
        """
        next_x, next_y = self.curr_x, self.curr_y
        if action == 'terminate':
            # In this case we are not discovering options we are just talking about a general MDP.
            if self.reward_function is None:
                if next_x == self.goal[0] and next_y == self.goal[1]:
                    return -1, -1  # absorbing state
                else:
                    return self.curr_x, self.curr_y

            # Otherwise we are talking about option discovery
            else:
                return -1, -1  # absorbing state

        if not self.is_wall(self.curr_x, self.curr_y):
            if action == 'up' and self.curr_x > 0:
                next_x, next_y = self.curr_x - 1, self.curr_y
            elif action == 'right' and self.curr_y < self.num_cols - 1:
                next_x, next_y = self.curr_x, self.curr_y + 1
            elif action == 'down' and self.curr_x < self.num_rows - 1:
                next_x, next_y = self.curr_x + 1, self.curr_y
            elif action == 'left' and self.curr_y > 0:
                next_x, next_y = self.curr_x, self.curr_y - 1

        if not self.is_wall(next_x, next_y):
            return next_x, next_y
        else:
            return self.curr_x, self.curr_y

    def get_next_reward(self, curr_x, curr_y, action, next_x, next_y):
        """
        Returns the reward the agent will observe in state (cX, cY) => a => (nX, nY)
        """
        # General Reward Function: Handle Negative Rewards => If Applicable
        if self.reward_function is None and self.use_negative_rewards:
            if self.is_wall(next_x, next_y) or self.coord2id(next_x, next_y) == self.num_states:
                return 0
            else:
                return -1
        elif self.reward_function is None and not self.use_negative_rewards:
            if next_x == self.goal[0] and next_y == self.goal[1]:
                return 1
            else:
                return 0

        # Eigenpurpose Reward Function:
        curr_idx, next_idx = self.coord2id(curr_x, curr_y), self.coord2id(next_x, next_y)
        reward = self.reward_function[next_idx] - self.reward_function[curr_idx]
        return reward

    def get_next_state_reward(self, curr, action):
        """
        Return the next state and reward given the current state and action to perform.

        :return: Tuple of next state, reward.
        """
        # In case it is the absorbing state encoding end of an episode
        if curr == self.num_states:
            return curr, 0

        # Get Current State Index
        temp_x, temp_y = self.curr_x, self.curr_y
        self.curr_x, self.curr_y = self.id2coord(curr)

        # Get Next State/Reward Info
        if self.reward_function is None and self.is_terminal():
            next_idx, reward = self.num_states, 0
        else:
            next_x, next_y = self.get_next_state(action)
            if next_x != -1 and next_y != -1:  # If not absorbing state
                reward = self.get_next_reward(self.curr_x, self.curr_y, action, next_x, next_y)
                next_idx = self.coord2id(next_x, next_y)
            else:
                reward, next_idx = 0, self.num_states

        # Restore to Initial
        self.curr_x, self.curr_y = temp_x, temp_y
        return next_idx, reward

    def get_next_state_reward_option(self, curr, o_pi, action_set):
        """Execute option until it terminates. Return number of timesteps it took and the terminal state."""
        # In case it is the absorbing state encoding end of an episode
        if curr == self.num_states:
            return curr, 0

        # Save Original State
        curr_idx = self.get_curr()
        goal_idx = self.coord2id(self.goal[0], self.goal[1])
        temp_x, temp_y = self.curr_x, self.curr_y

        # Update
        self.curr_x, self.curr_y = self.id2coord(curr)

        # Next State
        accum_reward, next_idx = 0, curr
        a_terminate, next_action = len(action_set) - 1, o_pi[curr]

        # Check if Termination
        if curr == goal_idx:
            next_idx = self.num_states
        elif self.use_negative_rewards and next_action == a_terminate and not self.is_wall(self.curr_x, self.curr_y):
            accum_reward = -1
        elif not self.use_negative_rewards and next_action == a_terminate and not self.is_wall(self.curr_x, self.curr_y):
            accum_reward = 0

        # Loop till Option Terminated
        while next_action != a_terminate:
            next_action = o_pi[curr]
            self.curr_x, self.curr_y = self.id2coord(curr)
            if self.reward_function is None and self.is_terminal():
                next_idx, next_action, reward = self.num_states, a_terminate, 0
            else:
                next_x, next_y = self.get_next_state(action_set[next_action])
                # If not absorbing state
                if next_x != -1 and next_y != -1:
                    reward = self.get_next_reward(self.curr_x, self.curr_y, next_action, next_x, next_y)
                    next_idx = self.coord2id(next_x, next_y)
                else:
                    reward, next_idx = 0, self.num_states

            accum_reward += reward
            curr = next_idx

        # Reset
        self.curr_x, self.curr_y = temp_x, temp_y
        return next_idx, accum_reward

