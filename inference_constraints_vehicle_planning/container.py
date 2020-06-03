'''
    for idx in range(max_steps):
        conflict = False
        # Check if it is the solution:
        for i in range(N):
            queen_pos = solution[i] # The row of the queen.
            for j in range(i + 1, N): # Checking horizontally.
                if solution[j] == queen_pos: # Stop the checking if there is a conflict.
                    conflict = True
                    break
            for k in range(1, N):
                if queen_pos + k < N and i + k < N: # Lower Diagonal
                    if solution[i + k] == queen_pos + k:
                        conflict = True
                        break
                if queen_pos - k >= 0 and i + k < N: # Upper Diagonal
                    if solution[i + k] == queen_pos - k:
                        conflict = True
                        break
        if conflict == False:
            return solution, idx
        # Modify board for best solution:
        else:
            # Get the conflicts per square:
            board = np.zeros((N, N))  # The board
            for i in range(N):
                queen_pos = solution[i]  # The row of the queen.
                for j in range(i + 1, N):  # Checking horizontally.
                    board[queen_pos, j] += 1
                for invj in range(0, i):
                    board[queen_pos, invj] += 1
                for k in range(1, N):
                    if queen_pos + k < N and i + k < N:  # Lower Diagonal
                        board[queen_pos + k, i + k] += 1
                    if queen_pos - k >= 0 and i + k < N:  # Upper Diagonal
                        board[queen_pos - k, i + k] += 1
                    if queen_pos + k < N and i - k >= 0:  # Reverse Lower Diagonal
                        board[queen_pos + k, i - k] += 1
                    if queen_pos - k >= 0  and i - k >= 0:  # Reverse Lower Diagonal
                        board[queen_pos - k, i - k] += 1
            # Select a random column:
            random_col = np.random.randint(0, N)
            # Select a random row with the lowest cost to move to:
            min_index = np.array(np.where(board[:,random_col] == board[:,random_col].min()))[0]
            min_index = np.random.choice(min_index, 1)
            solution[random_col] = min_index
            print(solution)
'''

'''
"""
    Problem 3 Template file
"""
import random
import math

import numpy as np

"""
Problem Statement
--------------------
Implement the planning algorithm called Rapidly-Exploring Random Trees (RRT)
for a problem setup given by the RRT_DUBINS_PROBLEM class.

INSTRUCTIONS
--------------------
1. The only file to be submitted is this file rrt_planning.py. Your implementation
   can be tested by running RRT_DUBINS_PROBLEM.PY (check the main function).
2. Read all class and function documentation in RRT_DUBINS_PROBLEM carefully.
   There are plenty of helper function in the class for ease of implementation.
3. Your solution must meet all the conditions specificed below.
4. Below are some do's and don'ts for this problem.

Conditions
-------------------
There are some conditions to be satisfied for an acceptable solution.
These may or may not be verified by the marking script.

1. Solution loop must not run for more that a certain number of random points
   (Specified by a class member called MAX_ITER). This is mainly a safety
   measure to avoid time-out related issues and will be generously set.
2. The planning function must return a list of nodes that represent a collision free path
   from start node to the goal node. The path states (path_x, path_y, path_yaw)
   specified by each node must be a dubins-style path and traverse from node i-1 -> node i.
   (READ the documentation of the node to understand the terminology)
3. The returned path should have the start node at index 0 and goal node at index -1,
   while parent node for node i from the list should be node i-1 from the list, ie,
   the path should be a valid list of nodes with dubin-style path connecting the nodes.
   (READ the documentation of the node to understand the terminology)
4. The node locations must not lie outside the map boundaries specified by
   RRT_DUBINS_PROBLEM.map_area

DO(s) and DONT(s)
-------------------
1. Rename the file to rrt_planning.py for submission.
2. Do not change change the PLANNING function signature.
3. Do not import anything other than what is already imported in this file.
4. You can write more function in this file in order to reduce code repitition
   but these function can only be used inside the PLANNING function.
   (since only the planning function will be imported)
"""

def planning(rrt_dubins, display_map=False):
    """
        Execute RRT planning using dubins-style paths. Make sure to populate the node_lis

        Inputs
        -------------
        rrt_dubins  - (RRT_DUBINS_PROBLEM) Class conatining the planning
                      problem specification
        display_map - (boolean) flag for animation on or off (OPTIONAL)

        Outputs
        --------------
        (list of nodes) This must be a valid list of connected nodes that form
                        a path from start to goal node

        NOTE: In order for rrt_dubins.draw_graph function to work properly, it is important
        to populate rrt_dubins.nodes_list with all valid RRT nodes.
    """
    # Fix Random Number Generator seed.
    random.seed(1)

    parent_node = rrt_dubins.start

    # LOOP for max iterations
    i = 0
    while i < rrt_dubins.max_iter:
        i += 1

        # Generate a random vehicle state (x, y, yaw).
        x_init = np.random.rand() * 2
        y_init = np.random.rand() * 4 - np.random.rand() * 2
        yaw_init = np.random.rand() * math.pi - np.random.rand() * -math.pi
        vehicle = rrt_dubins.Node(parent_node.x + x_init, parent_node.y + y_init, parent_node.yaw + yaw_init)

        #print(vehicle.x, vehicle.y, vehicle.yaw)

        # Find an existing path to the vehicle:
        min_cost = float("inf")
        min_node = None
        min_node_index = -1
        for i, node in enumerate(rrt_dubins.node_list):
            node_cost = rrt_dubins.calc_new_cost(node, vehicle)
            if node_cost < min_cost:
                min_cost = node_cost
                min_node = node
                min_node_index = i
        new_vehicle = rrt_dubins.propogate(min_node, vehicle)
        #print(new_node.x, new_node.y, new_node.yaw)

        # Check if the node between parent and vehicle has obstacle collision.
        # Add the path to nodes_list if it is valid:
        if rrt_dubins.check_collision(new_vehicle) and new_vehicle is not None:

            rrt_dubins.node_list = rrt_dubins.node_list[:min_node_index + 1]
            rrt_dubins.node_list.append(new_vehicle)  # Storing all valid paths
            goal_node = rrt_dubins.propogate(new_vehicle, rrt_dubins.goal) # Get the goal node.
            parent_node = new_vehicle

            if rrt_dubins.check_collision(goal_node):

                rrt_dubins.node_list.append(goal_node)
                # node to goal.parent contains the previous node
                #prev_node = goal_node.parent
                # continue to do this and add all the parent nodes to your return list that will be your path
                #while prev_node != None:
                    #rrt_dubins.node_list.append(prev_node)
                    #prev_node = prev_node.parent
                return rrt_dubins.node_list

        # Draw current view of the map
        # PRESS ESCAPE TO EXIT
        if display_map:
            rrt_dubins.draw_graph()

    if i == rrt_dubins.max_iter:
        print('reached max iterations')

        # Check if new_node is close to goal
        if True:
            print("Iters:", i, ", number of nodes:", len(rrt_dubins.node_list))
            break

    # Return path, which is a list of nodes leading to the goal

'''