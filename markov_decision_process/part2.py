# part2.py: Project 4 Part 2 script
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
  - Complete the policy_iteration method below
  - Please write abundant comments and write neat code
  - You can write any number of local functions
  - More implementation details in the Function comments
"""


def policy_iteration(env: mdp_env, agent: mdp_agent, max_iter = 1000) -> np.ndarray:
    """
    policy_iteration method implements VALUE ITERATION MDP solver,
    shown in AIMA (4ed pg 657). The goal is to produce an optimal policy
    for any given mdp environment.

    Inputs-
        agent: The MDP solving agent (mdp_agent)
        env:   The MDP environment (mdp_env)
        max_iter: Max iterations for the algorithm

    Outputs -
        policy: A list/array of actions for each state
                (Terminal states can have any random action)
       <agent>  Implicitly, you are populating the utlity matrix of
                the mdp agent. Do not return this function.
    """
    np.random.seed(1)  # TODO: Remove this

    policy = np.transpose(np.random.randint(len(env.actions), size=(len(env.states), 1)))[0]
    agent.utility = np.zeros([len(env.states), 1])

    ## START: Student code
    gamma = agent.gamma
    rewards = env.rewards
    trans = env.transition_model
    actions = env.actions
    states = env.states

    while max_iter > 0:
        for i in range(15):  # 15 cycles for policy evaluation.
            for state in states:  # Calculating for all possible states.
                possible_actions = trans[state].T  # The transitional model matrix.
                # Calculating the probability of a particular state being reached given the selected state wrt a particular act in policy.
                pos_act = [possible_actions[policy[state], s] * agent.utility[s] for s in range(len(possible_actions[policy[state]]))]
                agent.utility[state] = rewards[state] + gamma * sum(pos_act)  # Gives the agent's utility for each state.
        unchanged = True  # Termination variable: stops while loop if the value is False.
        for state in states:  # Going through all possible states.
            trans_utility = []  # Calculated utilities list.
            possible_actions = trans[state].T  # The transitional model matrix.
            for action in actions:  # Going through each action.
                # Calculating the probability of a particular state being reached given the selected state wrt a particular action.
                pos_act = [possible_actions[action, s] * agent.utility[s] for s in range(len(possible_actions[action]))]
                trans_utility.append(sum(pos_act))  # Sum of the probabilities of all states w/ a particular action.
            # Calculating the sum of probabilities of a particular state being reached given the selected state wrt a particular act in policy.
            pos_act_sum = sum([possible_actions[policy[state], s] * agent.utility[s] for s in range(len(possible_actions[policy[state]]))])
            if max(trans_utility) > pos_act_sum:  # If the state's max utility is greater than the current utility.
                policy[state] = np.argmax(trans_utility)  # Updates the particular act in final policy
                unchanged = False  # Changes the termination variable to false.
        if unchanged == True:
            break
    return policy
    ## END: Student code
