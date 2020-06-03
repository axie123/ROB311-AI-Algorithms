import numpy as np
### WARNING: DO NOT CHANGE THE NAME OF THIS FILE, ITS FUNCTION SIGNATURE OR IMPORT STATEMENTS


def min_conflicts_n_queens(initialization: list) -> (list, int):
    """
    Solve the N-queens problem with no conflicts (i.e. each row, column, and diagonal contains at most 1 queen).
    Given an initialization for the N-queens problem, which may contain conflicts, this function uses the min-conflicts
    heuristic(see AIMA, pg. 221) to produce a conflict-free solution.

    Be sure to break 'ties' (in terms of minimial conflicts produced by a placement in a row) randomly.
    You should have a hard limit of 1000 steps, as your algorithm should be able to find a solution in far fewer (this
    is assuming you implemented initialize_greedy_n_queens.py correctly).

    Return the solution and the number of steps taken as a tuple. You will only be graded on the solution, but the
    number of steps is useful for your debugging and learning. If this algorithm and your initialization algorithm are
    implemented correctly, you should only take an average of 50 steps for values of N up to 1e6.

    As usual, do not change the import statements at the top of the file. You may import your initialize_greedy_n_queens
    function for testing on your machine, but it will be removed on the autograder (our test script will import both of
    your functions).

    On failure to find a solution after 1000 steps, return the tuple ([], -1).

    :param initialization: numpy array of shape (N,) where the i-th entry is the row of the queen in the ith column (may
                           contain conflicts)

    :return: solution - numpy array of shape (N,) containing a-conflict free assignment of queens (i-th entry represents
    the row of the i-th column, indexed from 0 to N-1)
             num_steps - number of steps (i.e. reassignment of 1 queen's position) required to find the solution.
    """
    np.random.seed(100)
    N = len(initialization)
    solution = initialization.copy()
    num_steps = 0
    max_steps = 1000
    board = np.zeros((N, N))  # The board

    # Getting the initial conflicts per square of the current configuration:
    for i in range(N):  # For each column of the chessboard:
        queen_pos = solution[i]   # Selects of the queen's row of that column.
        for j in range(i + 1, N):   # Adds conflict to the right horizontally.
            board[queen_pos, j] += 1

        for invj in range(i - 1, -1, -1):  # Adds conflict to the left horizontally.
            board[queen_pos, invj] += 1

        for k in range(1, N):
            if queen_pos + k < N and i + k < N:  # Adds conflict to the lower right diagonal attack angle of the queen.
                board[queen_pos + k, i + k] += 1
            if queen_pos - k >= 0 and i + k < N:  # Adds conflict to the upper right diagonal attack angle of the queen.
                board[queen_pos - k, i + k] += 1
            if queen_pos + k < N and i - k >= 0:  # Adds conflict to the lower left diagonal attack angle of the queen.
                board[queen_pos + k, i - k] += 1
            if queen_pos - k >= 0 and i - k >= 0:  # Adds conflict to the upper left diagonal attack angle of the queen.
                board[queen_pos - k, i - k] += 1


    for idx in range(max_steps):
        conflict = False
        conflict_queens = []

        for c in range(len(solution)):  # Keeps track of the queens that have a conflict.
            if board[solution[c], c] != 0:
                conflict_queens += [c]
        if len(conflict_queens) != 0:
            conflict = True
        if conflict == False:  # Return the configuration if there is no conflict.
            return solution, idx
        else:
            # Select a random column with a conflict:
            random_col = np.random.choice(conflict_queens, 1)[0]
            # Keep track of the old row with the conflict:
            old_row = solution[random_col]
            # Select a random row with the lowest cost to move to in the column:
            min_index = np.array(np.where(board[:, random_col] == board[:, random_col].min()))[0]
            new_row = np.random.choice(min_index, 1)[0]
            # Diminish conflicts of the old position/row in the column:
            for j in range(random_col + 1, N):  # Diminish conflict to the right horizontally.
                board[old_row, j] -= 1

            for invj in range(random_col - 1, -1, -1):  # Diminish conflict to the left horizontally.
                board[old_row, invj] -= 1

            for k in range(1, N):
                if old_row + k < N and random_col + k < N:  # Diminish conflict to the lower right diagonal attack angle of the queen.
                    board[old_row + k, random_col + k] -= 1
                if old_row - k >= 0 and random_col + k < N:  # Diminish conflict to the upper right diagonal attack angle of the queen.
                    board[old_row - k, random_col + k] -= 1
                if old_row + k < N and random_col - k >= 0:  # Diminish conflict to the lower left diagonal attack angle of the queen.
                    board[old_row + k, random_col - k] -= 1
                if old_row - k >= 0 and random_col - k >= 0:  # Diminish conflict to the upper left diagonal attack angle of the queen.
                    board[old_row - k, random_col - k] -= 1

            # Adding conflicts of the new position/row in the column:
            solution[random_col] = new_row
            for j in range(random_col + 1, N):  # Adds conflict to the left horizontally.
                board[new_row, j] += 1
            for invj in range(random_col - 1, -1, -1):  # Adds conflict to the left horizontally.
                board[new_row, invj] += 1
            for k in range(1, N):
                if new_row + k < N and random_col + k < N:  # Adds conflict to the lower right diagonal attack angle of the queen.
                    board[new_row + k, random_col + k] += 1
                if new_row - k >= 0 and random_col + k < N:  # Adds conflict to the upper right diagonal attack angle of the queen.
                    board[new_row - k, random_col + k] += 1
                if new_row + k < N and random_col - k >= 0:  # Adds conflict to the lower left diagonal attack angle of the queen.
                    board[new_row + k, random_col - k] += 1
                if new_row - k >= 0 and random_col - k >= 0:  # Adds conflict to the upper left diagonal attack angle of the queen.
                    board[new_row - k, random_col - k] += 1
    print("No Solution Found")
    return solution, num_steps


if __name__ == '__main__':
    # Test your code here!
    from initialize_greedy_n_queens import initialize_greedy_n_queens
    from support import plot_n_queens_solution

    N = 1000
    # Use this after implementing initialize_greedy_n_queens.py
    assignment_initial = initialize_greedy_n_queens(N)

    # Plot the initial greedy assignment
    plot_n_queens_solution(assignment_initial)

    assignment_solved, n_steps = min_conflicts_n_queens(assignment_initial)

    # Plot the solution produced by your algorithm
    plot_n_queens_solution(assignment_solved)


