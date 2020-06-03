import random
import gym
import math
import numpy as np

"""
Agent Description:

TODO: Insert a 10-15 line description (80 characters wide) of the algorithm
that you implemented in your agent. If you used a reference paper or book,
you may add additional lines to cite this reference.
"""

env = gym.make('CartPole-v0')

class CartPoleAgent:

    def __init__(self, observation_space, action_space):
        #----- TODO: Add your code here. -----

        # Store observation space and action space.
        self.observation_space = observation_space
        self.action_space = action_space

        self.obs_state = (1, 1, 6, 3)
        self.lr = 0.1
        self.discount = 0.99
        self.epsilon = 0.1
        self.episode = -1

        self.act_table = np.zeros(self.obs_state + (self.action_space.n, ))

    def process_action(self, state):
        upper_state_b = self.observation_space.high
        lower_state_b = self.observation_space.low

        upper_state_b[1] = 1.5
        lower_state_b[1] = -1.5
        upper_state_b[3] = math.radians(50)
        lower_state_b[3] = -math.radians(50)

        w = [upper_state_b[i] - lower_state_b[i] for i in range(len(state))]
        r = [(state[i] - lower_state_b[i]) / w[i] for i in range(len(state))]
        bucket_indices = [int(round(r[i] * (self.obs_state[i] - 1))) for i in range(len(state))]

        # making the range of indices to [0, bucket_length]
        obs_state_converted = tuple([max(0, min(bucket_indices[i], self.obs_state[i] - 1)) for i in range(len(state))])

        return obs_state_converted

    def new_lr(self):
        new_lr = max(self.lr, min(1, 1 - math.log10((self.episode + 1) / 25)))
        return new_lr

    def new_epsilon(self):
        new_eps = max(self.epsilon, min(0.5, 1 - math.log10((self.episode + 1) / 25)))
        return new_eps

    def action(self, state):
        """Choose an action from set of possible actions."""
        #----- TODO: Add your code here. -----

        c_state = self.process_action(state)
        epsilon = self.new_epsilon()
        # Dummy agent just takes random actions...
        if random.random() < epsilon:
            action = self.action_space.sample()
        else:
            action = np.argmax(self.act_table[c_state])
        #print("Chose action " + str(action))
        return action

    def reset(self):
        """Reset the agent, if desired."""
        #----- TODO: Add your code here. -----
        self.episode += 1
        pass

    def update(self, state, action, reward, state_next, terminal):
        """Update the agent internally after an action is taken."""
        # ----- TODO: Add your code here. -----
        next_state = self.process_action(state_next)
        prev_state = self.process_action(state)
        lr = self.new_lr()

        max_Q_t1 = np.max(self.act_table[next_state])
        self.act_table[prev_state][action] += lr * (reward + self.discount * max_Q_t1 - self.act_table[prev_state][action])
