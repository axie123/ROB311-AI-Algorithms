import queue
import numpy as np
from search_problems import Node, GridSearchProblem, get_random_grid_problem



def a_star_search(problem):
    """
    Uses the A* algorithm to solve an instance of GridSearchProblem. Use the methods of GridSearchProblem along with
    structures and functions from the allowed imports (see above) to implement A*.

    :param problem: an instance of GridSearchProblem to solve
    :return: path: a list of states (ints) describing the path from problem.init_state to problem.goal_state[0]
             num_nodes_expanded: number of nodes expanded by your search
             max_frontier_size: maximum frontier size during search
    """
    ####
    #   COMPLETE THIS CODE
    ####
    start = problem.init_state # The position which we start at.
    finish = problem.goal_states # The declared position which we want to build a path to.
    M = problem.M # Dimensions of the map.
    N = problem.N
    grid_map = problem.grid_map
    cost_state = {} # A hash table to keep track of the total costs of all the states.
    parent_child = {} # Keeps track of the all the parents and their childs.

    num_nodes_expanded = 0
    max_frontier_size = 0
    path = [] # Collects edges of explored paths as tuples.
    priority_queue = queue.PriorityQueue() # Sets up a priority queue to hold the frontier.
    explored = set() # Creates a set to keep all the states that have been visited.
    found = False

    # Initialize start node:
    g_init = problem.manhattan_heuristic(start, start) # Calculates the manhattan dist from the start to itself (0)
    h_init = problem.heuristic(start) # Get's the estimated Euclidean distance from the starting node
    priority_queue.put((g_init + h_init, start)) # Adds the tuple to the priority queue sorted by the total cost.
    explored.add(start) # The starting node gets added to the explored list.
    cost_state[start] = 0 # The initial cost is going to be 0.
    parent_child[start] = start # The starting node is it's own parent.

    # Starts the A* search:
    while not priority_queue.empty() and found == False:
        max_frontier_size = max(priority_queue.qsize(), max_frontier_size) # Updates for the max frontier size.

        current_state = priority_queue.get() # Gets the node with the least cost from the queue.
        adj = problem.get_actions(current_state[1]) # Gets of the adj nodes of the priority node.
        for node in adj:
            h_to_final = problem.heuristic(node[1])  # The euclidean distance from the frontier node (adj node) to the final state.
            f = (cost_state[current_state[1]] + 1) + h_to_final - problem.heuristic(current_state[1]) # The total heuristic cost of the node.
            if cost_state.get(node[1]) and cost_state[node[1]] > f: # If a cost of the node already exist and it's greater than the newly calculate one:
                    cost_state[node[1]] = f # Update total cost of the node.
                    parent_child[current_state[1]] = node[1] # Update parent of the node.
            else: # Node doesn't exist as a cost: add the new cost into the cost-state hash.
                cost_state[node[1]] = f
                parent_child[current_state[1]] = node[1]
            if node[1] not in explored: # If the node is not visited:
                priority_queue.put((f, node[1])) # Put it into the priority queue to be examined for a path from it.
                explored.add(node[1]) # Visit the node.
                path.append(node) # Load the path from its parent to it.
            if node[1] == finish[0]: # Stop while and for loop if the final state is found.
                found = True
                break

    if not path: # If the path list is empty: there is no solution.
        return -1

    # Backtracking
    opt_path = []
    dest = path.pop(-1) # Get the tuple with the final state.
    opt_path.append(dest[0]) # Load the second last state.
    opt_path.append(dest[1]) # Load the goal state.
    back = dest[0] # Set a value to backtracking the edges.
    while path:
        edge = path.pop(-1) # Pop the next edge off the back of the collected edges.
        if edge[1] == back: # If the edge is connected to the backtracking node: add the node on the other end to the path.
            opt_path.insert(0, edge[0])
            back = edge[0]
    path = opt_path # Sets the final path.

    num_nodes_expanded = len(explored)

    return path, num_nodes_expanded, max_frontier_size


def search_phase_transition():
    """
    Simply fill in the prob. of occupancy values for the 'phase transition' and peak nodes expanded within 0.05. You do
    NOT need to submit your code that determines the values here: that should b computed on your own machine. Simply
    fill in the values!

    :return: tuple containing (transition_start_probability, transition_end_probability, peak_probability)
    """
    ####
    #   REPLACE THESE VALUES
    ####
    transition_start_probability = 0.3
    transition_end_probability = 0.5
    peak_nodes_expanded_probability = 0.4
    return transition_start_probability, transition_end_probability, peak_nodes_expanded_probability


if __name__ == '__main__':
    # Test your code here!
    # Create a random instance of GridSearchProblem
    p_occ = 0.25
    M = 200
    N = 200
    problem = get_random_grid_problem(p_occ, M, N)
    # Solve it
    path, num_nodes_expanded, max_frontier_size = a_star_search(problem)
    # Check the result
    correct = problem.check_solution(path)
    print("Solution is correct: {:}".format(correct))
    # Plot the result
    problem.plot_solution(path)

    # Experiment and compare with BFS