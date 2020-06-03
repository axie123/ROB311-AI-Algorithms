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
    path1 = [[] for i in vertices]
    path2 = [[] for i in vertices]

    frontier_1d = deque([])
    frontier_2d = deque([])
    explored1 = deque([])
    explored2 = deque([])

    # Upload the initial node (one direction):
    frontier_1d.append(start)
    explored1.append(start)
    path1[start].append(start)

    # Upload the final state (other direction):
    frontier_2d.append(finish[0])
    explored2.append(finish[0])
    path2[finish[0]].append(finish[0])

    while frontier_1d and frontier_2d:
        if max(len(frontier_1d),len(frontier_2d)) > max_frontier_size:
            max_frontier_size = max(len(frontier_1d),len(frontier_2d))

        current_node_1 = frontier_1d.popleft()
        #explored1.append(current_node_1)
        adj_1 = problem.get_actions(current_node_1)
        for node in adj_1:
            if node[1] not in explored1 and node[1] not in frontier_1d:
                frontier_1d.append(node[1])
                print("fw: ", frontier_1d)
                explored1.append(node[1])
                path1[node[1]] = path1[current_node_1] + [node[1]]

        if set(frontier_1d) & set(explored2):
            index = (set(explored1) & set(explored2)).pop()
            break

        current_node_2 = frontier_2d.popleft()
        #explored2.append(current_node_2)
        adj_2 = problem.get_actions(current_node_2)
        for node in adj_2:
            if node[1] not in explored2 and node[1] not in frontier_2d:
                frontier_2d.append(node[1])
                print("bw: ", frontier_2d)
                explored2.append(node[1])
                path2[node[1]] = path2[current_node_2] + [node[1]]

        if set(frontier_2d) & set(explored1):
            index = (set(explored1) & set(explored2)).pop()
            break

    path2[index].pop()
    path = path1[index] + path2[index][::-1]

    num_nodes_expanded = len(explored1) + len(explored2)

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