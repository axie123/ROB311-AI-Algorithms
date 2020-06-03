import random
import gym
import math
import numpy as np

"""
Agent Description: (I was told that 15-20 lines is ok. I have 17 lines).

The cartpole agent is based on a model-free reinforcement learning algorithm called Q-Learning.

The algorithm requires the use of a Q-table, which keeps track of the quality values of the agent. 
The Q-table in this agent includes a series of tables, each with two columns for each possible 
action, and is in 4-D. Each dimension is dedicated to a specific parameter out of the 4. The dim 
of each parameter is the amount of possible q-states and values that can be given for a particular 
action for that parameter. This would increase the search variety of our agent to give a higher 
chance for the optimal value to be chosen. In this case, a heavy emphasis is put on the pole angle 
and angular velocity, being at 6 and 3 respectively.

Every time the agent takes a step in an episode, an action would be selected based on the action 
that has the highest q-value utility given a state who's indices are scaled to fit the dimensions 
of the Q-table. A random value between 0 and 1 is generated to determine the amount of exploration 
that the agent is currently undertaken. If it is lower than the exploration constant, then a random 
action will be taken to explore more. If it's greater, then the action with the largest q-value 
from it's q-state for the parameter will be chosen to approach that optimal state. The agent would 
then take a step and update. The Q-state equation is used with the exploration rate and learning 
rate to update the Q-value of the action for the previous state accordingly. The learning rate and 
exploration rate is updated at each episode to make sure that over-fitting/toggling doesn't happen 
as the agent trains more.
"""

env = gym.make('CartPole-v0')

class CartPoleAgent:

    def __init__(self, observation_space, action_space):
        #----- TODO: Add your code here. -----

        # Store observation space and action space.
        self.observation_space = observation_space
        self.action_space = action_space

        self.obs_state = (1, 1, 6, 3)  # The observation state includes 4 parameters and their # of possible q-states:
        # 1. Cart Position - 1 possible q-state, 2. Cart Velocity - 1 possible q-state,
        # 3. Pole Angle - 6 possible q-states, 4. Angular Velocity - 3 possible q-states.

        self.upper_state_b = observation_space.high  # The upper limit for the parameters of the observation space.
        self.lower_state_b = observation_space.low  # The lower limit for the parameters of the observation space.
        self.upper_state_b[1] = 1.5  # Setting the upper bound of the cart velocity to be 1.5.
        self.lower_state_b[1] = -1.5  # Setting the lower bound of the cart velocity to be -1.5.
        self.upper_state_b[3] = math.radians(20)  # Setting the upper bound of angular velocity of pole to 20 rads.
        self.lower_state_b[3] = -math.radians(20)  # Setting the lower bound of angular velocity of pole to -20 rads.

        self.lr = 0.2  # Default learning rate set to 0.2.
        self.discount = 0.999  # Discount/gamma value set to 0.999.
        self.explore = 0.015  # Exploration value to 0.015.
        self.episode = -1  # Keeps track of the episode number to adjust the exploration value and the learning rate.

        # The Q-table to keep the states.
        self.act_table = np.zeros(self.obs_state + (self.action_space.n, ))

    # Scales the parameter indices of the given state for the constraints of the Q-table.
    def process_state(self, state):
        rec_states = []
        for i in range(len(state)):  # Given a state: makes sure that the parameters of the state are within bounds.
            if state[i] <= self.lower_state_b[i]:
                para_i = 0
            elif state[i] >= self.upper_state_b[i]:
                para_i = self.obs_state[i] - 1
            else:  # For parameters within bounds:
                w = self.upper_state_b[i] - self.lower_state_b[i]  # Setting the weight of the parameter index to be the range of its boundary for relative scaling.
                low_scaling = (self.obs_state[i] - 1) * self.lower_state_b[i] / w  # Scaling the parameter index to fit the lower bounds of the agent's parameter table.
                scaling_factor = (self.obs_state[i] - 1) / w  # Scaling to fit the parameter index with the agent's parameter table relative to its original scaling.
                para_i = int(round(scaling_factor * state[i] - low_scaling))  # Calculates the rescaled parameter index for the state.
            rec_states.append(para_i)
        action_ready = tuple(rec_states) # Turns the scaled states into a tuple.
        return action_ready

    # Updates to the appropriate learning rate to prevent toggling in later episodes.
    def new_lr(self):
        new_lr = max(self.lr, min(0.5, 1 - math.log10((self.episode + 1) / 30)))
        return new_lr

    # Updates to the appropriate exploration rate (explore) to make sure that
    # unnecessary exploration doesn't happen later on.
    def new_explore(self):
        new_explore = max(self.explore, min(0.5, 1 - math.log10((self.episode + 1) / 30)))
        return new_explore

    # Pick an action:
    def action(self, state):
        """Choose an action from set of possible actions."""
        #----- TODO: Add your code here. -----

        c_state = self.process_state(state)
        explore = self.new_explore()
        if random.random() < explore:  # Random guess on amount of exploration done already.
            action = self.action_space.sample()  # Select a random action if the agent didn't explore enough to explore more.
        else:
            action = np.argmax(self.act_table[c_state])  # The action with the maximum value will be selected to exploit the information gained.
        return action

    # The reset isn't going to permanently reset the q-table and parameters, but keeps track of the episode.
    def reset(self):
        """Reset the agent, if desired."""
        #----- TODO: Add your code here. -----
        self.episode += 1
        pass

    # Updates the knowledge of the agent.
    def update(self, state, action, reward, state_next, terminal):
        """Update the agent internally after an action is taken."""
        # ----- TODO: Add your code here. -----
        next_state = self.process_state(state_next)  # Represents the next state of the agent.
        prev_state = self.process_state(state)  # Represents the current state of the agent.
        lr = self.new_lr()  # Updates the learning rate.

        max_Q_t1 = np.amax(self.act_table[next_state])  # Gets the maximum estimate of the future Q value if it's on the next state.
        if terminal:
            self.act_table[prev_state][action] += 0  # If the terminal state is reached, the neg reward given by the testing isn't added.
        else:
            # If the terminal state isn't reached, the Q-valuee for the action and state of the agent will be updated according to the Q-update equation.
            self.act_table[prev_state][action] += lr * (reward + self.discount * max_Q_t1 - self.act_table[prev_state][action])