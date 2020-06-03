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

    # Note: My original method was getting very convoluted and therefore wasn't fast enough.
    # I had some classmates help me understand the Node API that was given to me.

    start = problem.init_state  # The position which we start at.
    finish = problem.goal_states[0]  # The declared position which we want to build a path to.
    M = problem.M  # Dimensions of the map.
    N = problem.N
    grid_map = problem.grid_map

    num_nodes_expanded = 0
    max_frontier_size = 0
    path = [] # Collects edges of explored paths as tuples.
    priority_queue = queue.PriorityQueue() # Sets up a priority queue to hold the frontier.
    explored = set()

    g_init = problem.manhattan_heuristic(start, start) # Calculates the manhattan dist from the start to itself (= 0)
    h_init = problem.heuristic(start) # Get's the estimated Euclidean distance from the starting node
    priority_queue.put((g_init + h_init, Node(None, start, (start, start), 0))) # Adds the tuple to the priority queue sorted by the total cost and the node package.
    explored.add(start) # Creates a set to keep all the states that have been visited.
    found = False # State of whether the final state has been found.

    while not priority_queue.empty() and found == False:

        max_frontier_size = max(priority_queue.qsize(), max_frontier_size) # Gets the maximum size of the frontier.

        current_state = priority_queue.get() # Gets the node with the least cost from the queue.

        if current_state[1].state == finish: # Stop while and for loop if the final state is found.
            path = problem.trace_path(current_state[1], start) # Set the final path.
            found = True

        adj = problem.get_actions(current_state[1].state) # Gets of the adj nodes of the priority node.
        for node in adj:
            adj_node = problem.get_child_node(current_state[1], node) # Unpacks the information of the frontier node:
                                                                      # parent node, its name, its path, and the cost to get there.
            if adj_node.state not in explored: # If the frontier node is not visited:
                h_to_final = problem.heuristic(adj_node.state) # The euclidean distance from the frontier node (adj node) to the final state.
                g = adj_node.path_cost # Total cost to get to the frontier node.
                f = g + h_to_final # The total heuristic cost of the node.
                priority_queue.put((f, adj_node)) # Put it into the priority queue to be examined for a path from it.
                explored.add(adj_node.state) # Visit the frontier node.

    num_nodes_expanded = len(explored) # The number of expanded nodes = # of nodes touched.

    return path, num_nodes_expanded, max_frontier_size


def search_phase_transition():
    """
    Simply fill in the prob. of occupancy values for the 'phase transition' and peak nodes expanded within 0.05. You do
    NOT need to submit your code that determines the values here: that should be computed on your own machine. Simply
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
    M = 100
    N = 100
    problem = get_random_grid_problem(p_occ, M, N)
    # Solve it
    path, num_nodes_expanded, max_frontier_size = a_star_search(problem)
    # Check the result
    correct = problem.check_solution(path)
    print("Solution is correct: {:}".format(correct))
    # Plot the result
    problem.plot_solution(path)

    # Experiment and compare with BFS