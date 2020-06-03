from collections import deque
import numpy as np
from search_problems import Node, GraphSearchProblem

def bidirectional_search(problem):
    """
        Implement a bidirectional search algorithm that takes instances of SimpleSearchProblem (or its derived
        classes) and provides a valid and optimal path from the initial state to the goal state.

        :param problem: instance of SimpleSearchProblem
        :return: path: a list of states (ints) describing the path from problem.init_state to problem.goal_state[0]
                 num_nodes_expanded: number of nodes expanded by your search
                 max_frontier_size: maximum frontier size during search
        """
    ####
    #   COMPLETE THIS CODE
    ####
    # Note: I realized that my whole BFS strategy was wrong with this particular problem. Had to rebuild it from the ground up.

    vertices = problem.V # Loads all the vertices of the graph entered into the method.
    start = problem.init_state # The node which we start at.
    finish = problem.goal_states # The declared node which we want to build a path to.

    max_frontier_size = 0
    num_nodes_expanded = 0
    pathf = [[] for i in vertices] # Made a design choice to use a list of lists. Each nested list represents the path from the starting node to that node.
    pathb = [[] for i in vertices] # This list works the same way has the previous, except it starts from the final state.

    frontier_f = set() # Queue to keep all the nodes in the frontier coming from the start.
    frontier_b = set() # Queue to keep all the nodes in the frontier coming from the finish.
    exploredf = set() # Queue to keep all the nodes that we visted from the start.
    exploredb = set() # Queue to keep all the nodes that we visted from the finish.
    found = False # State of whether an intersection is found
    len_path = float("inf") # Keeps an upper bound on the length of the path.

    # Upload the initial node (one direction):
    frontier_f.add(start) # Want to start off by adding the start node to the frontier.
    exploredf.add(start) # The way how we do it, the start node is already visited.
    pathf[start].append(start) # The start node is added to the path.

    # Upload the final state (other direction):
    frontier_b.add(finish[0]) # Want to start off by adding the finish node to the frontier.
    exploredb.add(finish[0]) # The way how we do it, the finish node is already visited byt the other BFS.
    pathb[finish[0]].append(finish[0]) # The finish node is added to the path.

    while frontier_f and frontier_b and found == False:
        if max(len(frontier_f),len(frontier_b)) > max_frontier_size: # Updates the greatest frontier size.
            max_frontier_size = max(len(frontier_f),len(frontier_b))

        current_frontier_f = set()
        for f in frontier_f: # Checks every node in the current frontier of the forward BFS.
            adj_f = problem.get_actions(f) # Get all of the adjacent nodes for each.
            for adj_node in adj_f:
                if adj_node[1] not in exploredf and adj_node[1] not in frontier_f: # If the adjacent node is untouched by the forward BFS:
                    current_frontier_f.add(adj_node[1]) # Add the adj node to the frontier of the BFS
                    pathf[adj_node[1]] = pathf[f] + [adj_node[1]] # Add its path to the path list of lists for the BFS.
                    if adj_node[1] in exploredf:
                        pathb[adj_node[1]].pop() # We pop off the common node from the backwards BFS to create the final path.
                        curr_len = len(pathf[adj_node[1]] + pathb[adj_node[1]][::-1]) # Gets the len of the final path.
                        if curr_len < len_path: # Updates the path if the length is smaller than the current path length.
                            len_path = curr_len
                            path = pathf[adj_node[1]] + pathb[adj_node[1]][::-1]
                        found = True # Terminates the loop if the adjacent node is already explored.
                    exploredf.add(f) # Otherwise, add the frontier node as a visited node for the forward BFS.
        frontier_f = current_frontier_f # Updates the frontier layer by layer of the forward search.

        current_frontier_b = set()
        for b in frontier_b:  # Checks every node in the current frontier of the backwards BFS.
            adj_b = problem.get_actions(b) # Get all of the adjacent nodes for each.
            for adj_node in adj_b:
                if adj_node[1] not in exploredb and adj_node[1] not in frontier_b: # If the adjacent node is untouched by the backwards BFS:
                    current_frontier_b.add(adj_node[1]) # Add the adj node to the frontier of the backwards BFS
                    pathb[adj_node[1]] = pathb[b] + [adj_node[1]] # Add its path to the path list of lists for the backward BFS.
                    if adj_node[1] in exploredf: # If the node reached by the backwards BFS is visited by the forward BFS.
                        pathb[adj_node[1]].pop() # We pop off the common node from the backwards BFS to create the final path.
                        curr_len = len(pathf[adj_node[1]] + pathb[adj_node[1]][::-1]) # Gets the len of the final path.
                        if curr_len < len_path: # Updates the path if the length is smaller than the current path length.
                            len_path = curr_len
                            path = pathf[adj_node[1]] + pathb[adj_node[1]][::-1]
                        found = True # Terminates the loop if the adjacent node is already explored.
                    exploredb.add(b) # Otherwise, add the frontier node as a visited node for the backwards BFS.
        frontier_b = current_frontier_b # Updates the frontier layer by layer of the backwards search.

        num_nodes_expanded = len(exploredf) + len(exploredb) # The # of nodes expanded is the sum of the number of nodes explored by both BFS.

    return path, num_nodes_expanded, max_frontier_size

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
    path, num_nodes_expanded, max_frontier_size = bidirectional_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)

    # Use stanford_large_network_facebook_combined.txt to make your own test instances
    E = np.loadtxt('stanford_large_network_facebook_combined.txt', dtype=int)
    V = np.unique(E)
    goal_states = [349]
    init_state = 0
    problem = GraphSearchProblem(goal_states, init_state, V, E)
    path, num_nodes_expanded, max_frontier_size = bidirectional_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)


    # Be sure to compare with breadth_first_search!