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
    cost_state = {}
    parent_child = {}

    num_nodes_expanded = 0
    max_frontier_size = 0
    path = []
    priority_queue = queue.PriorityQueue()
    explored = set()
    found = False

    # Initialize start node:
    g_init = problem.manhattan_heuristic(start, start)
    h_init = problem.heuristic(start)
    priority_queue.put((g_init + h_init, start))
    explored.add(start)
    cost_state[start] = 0
    parent_child[start] = start

    while not priority_queue.empty() and found == False:
        if priority_queue.qsize() > max_frontier_size:
            max_frontier_size = priority_queue.qsize()
        current_state = priority_queue.get()
        adj = problem.get_actions(current_state[1])
        for node in adj:
            h_to_final = problem.heuristic(node[1])  # The euclidean distance from the frontier to the final state.
            f = cost_state[current_state[1]] + 1 + h_to_final - problem.heuristic(current_state[1])
            if cost_state.get(node[1]):
                if cost_state[node[1]] > f:
                    cost_state[node[1]] = f
                    parent_child[current_state[1]] = node[1]
            else:
                cost_state[node[1]] = f
                parent_child[current_state[1]] = node[1]
            if node[1] not in explored:
                priority_queue.put((f, node[1]))
                explored.add(node[1])
                path.append(node)
                if node[1] == finish[0]:
                    found = True
                    break
    if not path:
        return -1

    # Backtracking
    opt_path = []
    dest = path.pop(-1)
    opt_path.append(dest[0])
    opt_path.append(dest[1])
    back = dest[0]
    while path:
        edge = path.pop(-1)
        if edge[1] == back:
            opt_path.insert(0, edge[0])
            back = edge[0]
    path = opt_path

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