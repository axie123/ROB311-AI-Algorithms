# part1_2.py: Project 4 Part 1 script
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
from mdp_env import mdp_env
from mdp_agent import mdp_agent

## WARNING: DO NOT CHANGE THE NAME OF THIS FILE, ITS FUNCTION SIGNATURE OR IMPORT STATEMENTS

"""
INSTRUCTIONS
-----------------
  - Complete the value_iteration method below
  - Please write abundant comments and write neat code
  - You can write any number of local functions
  - More implementation details in the Function comments
"""

def value_iteration(env: mdp_env, agent: mdp_agent, eps: float, max_iter = 1000) -> np.ndarray:
    """
    value_iteration method implements VALUE ITERATION MDP solver,
    shown in AIMA (4ed pg 653). The goal is to produce an optimal policy
    for any given mdp environment.

    Inputs
    ---------------
        agent: The MDP solving agent (mdp_agent)
        env:   The MDP environment (mdp_env)
        eps:   Max error allowed in the utility of a state
        max_iter: Max iterations for the algorithm

    Outputs
    ---------------
        policy: A list/array of actions for each state
                (Terminal states can have any random action)
       <agent>  Implicitly, you are populating the utlity matrix of
                the mdp agent. Do not return this function.
    """
    # Got advice from a classmate (I was told it's ok).
    policy = np.empty_like(env.states)
    agent.utility = np.zeros([len(env.states), 1])

    ## START: Student code
    states = env.states
    actions = env.actions
    rewards = env.rewards
    discount = agent.gamma
    trans_model = env.transition_model

    while max_iter > 0:
        delta = 0
        U = agent.utility.copy()  # A copy of the utility for the agent.
        for state in states: # Going through each possible state.
            trans_utility = []  # Calculated utilities list.
            possible_actions = trans_model[state].T  # The transition model for the problem.
            for action in actions:  # Going through each action.
                # Calculating the probability of a particular state being reached given the selected state wrt a particular action.
                pos_act = [possible_actions[action, s]*U[s] for s in range(len(possible_actions[action]))]
                trans_utility.append(sum(pos_act))  # Sum of the probabilities of all states w/ a particular action.
            agent.utility[state] = rewards[state] + discount*max(trans_utility)  # Calculates the agent's utility value for a particular state.
            if abs(agent.utility[state] - U[state]) > delta:  # Updating the delta value if the abs utility difference is greater than it.
                delta = abs(agent.utility[state] - U[state])
        if delta < eps * (1 - discount) / discount:  # Terminating condition if delta val is smaller than the constant.
            break
        max_iter -= 1  # Decrements the step counter by 1.

    # Optimizing the policy:
    for p in range(len(policy)):  # Going through each act in the final policy.
        u = []
        possible_acts = trans_model[p].T  # The transition model for the act.
        for act in actions:  # Going through each possible action.
            # Checking if the max argument of the particular act is an integer.
            if agent.utility[np.argmax(possible_acts[act])] != None and type(agent.utility[np.argmax(possible_acts[act])]) == int:
                u.append(agent.utility[np.argmax(possible_acts[act])])  # Getting the utility with the max argument of the particular act in the transitional model.
            else:
                u.append(0)  # If the utility of the max argument is not available, upload 0 by default.
        policy[p] = np.argmax(u)  # The updated act in the final policy is the max utility calculated.

    ## END Student code
    return policy
