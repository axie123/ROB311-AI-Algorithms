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
    start = problem.init_state
    finish = problem.goal_states
    M = problem.M
    N = problem.N
    grid_map = problem.grid_map

    num_nodes_expanded = 0
    max_frontier_size = 0
    path = []
    priority_queue = queue.PriorityQueue()
    explored = []

    # Initialize start node:
    g_init = problem.manhattan_heuristic(start, start)
    h_init = problem.heuristic(start)
    priority_queue.put((g_init + h_init, start))
    explored.append(start)

    while not priority_queue.empty():
        if priority_queue.qsize() > max_frontier_size:
            max_frontier_size = priority_queue.qsize()
        current_state = priority_queue.get()
        if current_state[1] == finish[0]:
            break
        adj = problem.get_actions(current_state[1])
        for node in adj:
            g = problem.manhattan_heuristic(current_state[1], node[1])
            # g_prev = problem.manhattan_heuristic(start, current_state[1])
            g_prev = problem.heuristic(current_state[1])
            h = problem.heuristic(node[1])
            f = problem.manhattan_heuristic(start, current_state[1]) + g - g_prev
            if node[1] not in explored:
                priority_queue.put((f+h, node[1]))
                explored.append(node[1])
                path.append(node)
                if node[1] == finish[0]:
                    break
    # Backtracking
    opt_path = []
    dest = path.pop(-1)
    opt_path.append(dest[1])
    opt_path.append(dest[0])
    back = dest[0]
    while path:
        edge = path.pop(-1)
        if edge[1] == back:
            opt_path.append(edge[0])
            back = edge[0]
    path = opt_path[::-1]

    num_nodes_expanded = len(explored)

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
    transition_start_probability = -1.0
    transition_end_probability = -1.0
    peak_nodes_expanded_probability = -1.0
    return transition_start_probability, transition_end_probability, peak_nodes_expanded_probability


if __name__ == '__main__':
    # Test your code here!
    # Create a random instance of GridSearchProblem
    p_occ = 0.25
    M = 50
    N = 50
    problem = get_random_grid_problem(p_occ, M, N)
    # Solve it
    path, num_nodes_expanded, max_frontier_size = a_star_search(problem)
    # Check the result
    correct = problem.check_solution(path)
    print("Solution is correct: {:}".format(correct))
    # Plot the result
    problem.plot_solution(path)

    # Experiment and compare with BFS