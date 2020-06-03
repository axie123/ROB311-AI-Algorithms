from collections import deque
import numpy as np
from search_problems import Node, GraphSearchProblem

def breadth_first_search(problem):
    """
    Implement a simple breadth-first search algorithm that takes instances of SimpleSearchProblem (or its derived
    classes) and provides a valid and optimal path from the initial state to the goal state. Useful for testing your
    bidirectional and A* search algorithms.

    :param problem: instance of SimpleSearchProblem
    :return: path: a list of states (ints) describing the path from problem.init_state to problem.goal_state[0]
             num_nodes_expanded: number of nodes expanded by your search
             max_frontier_size: maximum frontier size during search
    """
    vertices = problem.V # Loads all the vertices of the graph entered into the method.
    edges = problem.E
    start = problem.init_state # The node which we start at.
    finish = problem.goal_states # The declared node which we want to build a path to.

    max_frontier_size = 0
    num_nodes_expanded = 0
    path = [[] for i in vertices] # Made a design choice to use a list of lists.
                                  # Each nested list represents the path from the starting node to that node.

    frontier = deque([]) # Queue to keep all the nodes in the frontier.
    explored = deque([]) # Queue to keep all the nodes that we visited.

    # Upload the initial node:
    frontier.append(start) # Want to start off by adding the start node to the frontier.
    explored.append(start) # The way how we do it, the start node is already visited.
    path[start].append(start) # The start node is added to the path.

    # Start BFS:
    while frontier: # Will loop through as long there are still nodes in the frontier.
        if len(frontier) > max_frontier_size: # This 'if' statement updates the frontier size.
            max_frontier_size = len(frontier)
        current_node = frontier.popleft()
        if current_node == finish: # Stops the BFS loop if the finish node is found.
            break
        adj = problem.get_actions(current_node) # Gets the adjacent nodes that would expand the frontier.
        for node in adj:
            if node[1] not in explored: # Checks if the new frontier node is already visited.
                frontier.append(node[1]) # Makes the new node part of the frontier queue.
                explored.append(node[1]) # Visits it.
                path[node[1]] = path[current_node] + [node[1]] # Adds the path from the previous frontier node to the new adj node.

    path_1 = path[current_node] # Gets the path from the start node to the node

    num_nodes_expanded = len(explored) # The # of nodes explored before reaching the final state.

    return path_1, num_nodes_expanded, max_frontier_size

if __name__ == '__main__':
    # Simple example
    goal_states = [0]
    init_state = 9
    V = np.arange(0, 10)
    E = np.array([[0, 1],
                  [1, 2],
                  [2, 3],
                  [3, 4],
                  [4, 5],
                  [5, 6],
                  [6, 7],
                  [7, 8],
                  [8, 9],
                  [0, 6],
                  [1, 7],
                  [2, 5],
                  [9, 4]])
    problem = GraphSearchProblem(goal_states, init_state, V, E)
    path, num_nodes_expanded, max_frontier_size = breadth_first_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)

    # Use stanford_large_network_facebook_combined.txt to make your own test instances
    E = np.loadtxt('stanford_large_network_facebook_combined.txt', dtype=int)
    V = np.unique(E)
    goal_states = [349]
    init_state = 0
    problem = GraphSearchProblem(goal_states, init_state, V, E)
    path, num_nodes_expanded, max_frontier_size = breadth_first_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)

