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

#### THIS IS NOT MY FINAL CODE!!!!

env = gym.make('CartPole-v0')

class CartPoleAgent:

    def __init__(self, observation_space, action_space):
        #----- TODO: Add your code here. -----

        # Store observation space and action space.
        self.observation_space = observation_space
        self.action_space = action_space

        self.upper_state_b = observation_space.high
        self.lower_state_b = observation_space.low
        self.upper_state_b[1] = 1.5
        self.lower_state_b[1] = -1.5
        self.upper_state_b[3] = math.radians(20)
        self.lower_state_b[3] = -math.radians(20)

        self.obs_state = (1, 1, 6, 3)
        self.lr = 0.2
        self.discount = 0.999
        self.epsilon = 0.01
        self.episode = -1

        self.act_table = np.zeros(self.obs_state + (self.action_space.n, ))

    def process_action(self, state):
        processed_states = []
        for i in range(len(state)):
            if state[i] <= self.lower_state_b[i]:
                bucket_index = 0
            elif state[i] >= self.upper_state_b[i]:
                bucket_index = self.obs_state[i] - 1
            else:
                w = self.upper_state_b[i] - self.lower_state_b[i]
                diff_low = (self.obs_state[i] - 1) * self.lower_state_b[i] / w
                scaling = (self.obs_state[i] - 1) / w
                bucket_index = int(round(scaling * state[i] - diff_low))
            processed_states.append(bucket_index)
        action_ready = tuple(processed_states)
        return action_ready

    def new_lr(self):
        new_lr = max(self.lr, min(0.5, 1 - math.log10((self.episode + 1) / 40)))
        return new_lr

    def new_epsilon(self):
        new_eps = max(self.epsilon, min(0.5, 1 - math.log10((self.episode + 1) / 40)))
        return new_eps

    def action(self, state):
        """Choose an action from set of possible actions."""
        #----- TODO: Add your code here. -----

        c_state = self.process_action(state)
        epsilon = self.new_epsilon()
        if random.random() < epsilon:
            action = self.action_space.sample()
        else:
            action = np.argmax(self.act_table[c_state])
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

        max_Q_t1 = np.amax(self.act_table[next_state]) # ReLu or Leaky ReLu
        if terminal:
            self.act_table[prev_state][action] += (lr * (reward + self.discount * max_Q_t1 - self.act_table[prev_state][action]))/100
        else:
            self.act_table[prev_state][action] += lr * (reward + self.discount * max_Q_t1 - self.act_table[prev_state][action])