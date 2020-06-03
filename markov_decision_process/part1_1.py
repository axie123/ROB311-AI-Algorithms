# part1_1.py: Project 4 Part 1 script
#
# --
# Artificial Intelligence
# ROB 311 Winter 2020
# Programming Project 4
#
# --
# University of Toronto Institute for Aerospace Studies
# Stars Lab
#
# Course Instructor:
# Dr. Jonathan Kelly
# jkelly@utias.utoronto.ca
#
# Teaching Assistant:
# Matthew Giamou
# mathhew.giamau@robotics.utias.utoronto.ca
#
# Abhinav Grover
# abhinav.grover@robotics.utias.utoronto.ca


###
# Imports
###

import numpy as np
from mdp_cleaning_task import cleaning_env

## WARNING: DO NOT CHANGE THE NAME OF THIS FILE, ITS FUNCTION SIGNATURE OR IMPORT STATEMENTS

"""
INSTRUCTIONS
-----------------
  - Complete the method get_transition_model which creates the
    transition probability matrix for the cleaning robot problem described in the
    project document.
  - Please write abundant comments and write neat code
  - You can write any number of local functions
  - More implementation details in the Function comments
"""

def get_transition_model(env: cleaning_env) -> np.ndarray:
    """
    get_transition_model method creates a table of size (SxSxA) that represents the
    probability of the agent going from s1 to s2 while taking action a
    e.g. P[s1,s2,a] = 0.5
    This is the method that will be used by the cleaning environment (described in the
    project document) for populating its transition probability table

    Inputs
    --------------
        env: The cleaning environment

    Outputs
    --------------
        P: Matrix of size (SxSxA) specifying all of the transition probabilities.
    """


    P = np.zeros([len(env.states), len(env.states), len(env.actions)])

    ## START: Student Code
    actions = env.actions
    states = env.states
    terminal = env.terminal

    for state in states:  # Iterating through all the possible states.
        if state not in terminal:  # Checking if the state is an intermediate state.
            left_state = state - 1  # Sets the position index of the state to the left of the current state.
            right_state = state + 1  # Sets the position index of the state to the right of the current state.
            for action in actions:  # Going through all possible actions.
                P[state, state, action] = 0.15  # Probabilities for remaining stationary.
                if action == 0:  # Probabilities for going left based on facing direction.
                    P[state, left_state, action] = 0.8  # Facing left.
                    P[state, right_state, action] = 0.05  # Facing right.
                elif action == 1:  # Probabilities for going right based on facing direction.
                    P[state, left_state, action] = 0.05  # Facing left.
                    P[state, right_state, action] = 0.8  # Facing right.

    ## END: Student code
    return P
