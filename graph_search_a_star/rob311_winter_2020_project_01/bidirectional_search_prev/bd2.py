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
    vertices = problem.V
    start = problem.init_state
    finish = problem.goal_states

    max_frontier_size = 0
    num_nodes_expanded = 0
    pathf = [[] for i in vertices]
    pathb = [[] for i in vertices]

    frontier_f = []
    frontier_b = []
    exploredf = []
    exploredb = []
    index = []
    found = False

    # Upload the initial node (one direction):
    frontier_f.append(start)
    exploredf.append(start)
    pathf[start].append(start)

    # Upload the final state (other direction):
    frontier_b.append(finish[0])
    exploredb.append(finish[0])
    pathb[finish[0]].append(finish[0])

    while frontier_f and frontier_b and found == False:

        current_frontier_f = []
        for f in range(len(frontier_f)): # Checks every node in the current frontier
            adj_f = problem.get_actions(frontier_f[f])
            for adj_node in adj_f:
                if adj_node[1] not in exploredf and adj_node[1] not in frontier_f:
                    current_frontier_f.append(adj_node[1])
                    #exploredf.append(adj_node[1]) # Loads the explored node into the explored
                    print("f: ", exploredf)
                    '''
                    if set(exploredf) and set(exploredb):
                        index = (set(exploredf) & set(exploredb))
                        #pathf[adj_node[1]] = pathf[frontier_f[f]] + [index.pop()]
                        found = True
                    '''
                    pathf[adj_node[1]] = pathf[frontier_f[f]] + [adj_node[1]]
                    exploredf.append(frontier_f[f])
        frontier_f = current_frontier_f

        current_frontier_b = []
        for b in range(len(frontier_b)):  # Checks every node in the current frontier
            adj_b = problem.get_actions(frontier_b[b])
            for adj_node in adj_b:
                if adj_node[1] not in exploredb and adj_node[1] not in frontier_b:
                    current_frontier_b.append(adj_node[1])
                    # exploredb.append(adj_node[1])  # Loads the explored node into the explored
                    print("b: ", exploredb)
                    if adj_node[1] in exploredf:
                        # index = (set(exploredf) & set(exploredb))
                        ##pathb[adj_node[1]] = pathb[frontier_b[b]] + [index.pop()]
                        found = True
                    pathb[adj_node[1]] = pathb[frontier_b[b]] + [adj_node[1]]
                    exploredb.append(frontier_b[b])
        frontier_b = current_frontier_b


    print(index)
    print(pathf)
    print(pathb)

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