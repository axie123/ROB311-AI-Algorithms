import numpy as np
### WARNING: DO NOT CHANGE THE NAME OF THIS FILE, ITS FUNCTION SIGNATURE OR IMPORT STATEMENTS


def initialize_greedy_n_queens(N: int) -> list:
    """
    This function takes an integer N and produces an initial assignment that greedily (in terms of minimizing conflicts)
    assigns the row for each successive column. Note that if placing the i-th column's queen in multiple row positions j
    produces the same minimal number of conflicts, then you must break the tie RANDOMLY! This strongly affects the
    algorithm's performance!

    Example:
    Input N = 4 might produce greedy_init = np.array([0, 3, 1, 2]), which represents the following "chessboard":

     _ _ _ _
    |Q|_|_|_|
    |_|_|Q|_|
    |_|_|_|Q|
    |_|Q|_|_|

    which has one diagonal conflict between its two rightmost columns.

    You many only use numpy, which is imported as np, for this question. Access all functions needed via this name (np)
    as any additional import statements will be removed by the autograder.

    :param N: integer representing the size of the NxN chessboard
    :return: numpy array of shape (N,) containing an initial solution using greedy min-conflicts (this may contain
    conflicts). The i-th entry's value j represents the row  given as 0 <= j < N.
    """
    greedy_init = np.zeros(N)  # Each index represents the column and the value represents the position.
    board = np.zeros((N, N))
    greedy_init[0] = np.random.randint(0, N)  # First queen goes in a random spot
    current_pos = int(greedy_init[0])  # The row that the initial queen is in.

    ### YOUR CODE GOES HERE
    for queen in range(0, N-1):  # Iterates through the columns of the chess board.
        for i in range(queen + 1,N):
            board[current_pos, i] += 1   # Adds a conflict of 1 to all the positions on the horizontal direction of attack of the queen.
        for j in range(1, N):
            if current_pos + j < N and queen + j < N:  # Adds a conflict of 1 to all the positions on the lower diagonal direction of attack of the queen.
                board[current_pos + j, queen + j] += 1
            if current_pos - j >= 0 and queen + j < N:  # Adds a conflict of 1 to all the positions on the upper diagonal direction of attack of the queen.
                board[current_pos - j, queen + j] += 1
        min_index = np.array(np.where(board[:,queen + 1] == board[:,queen + 1].min()))[0]  # Finds positions w/ minimal conflicts on the next column.
        min_index = np.random.choice(min_index, 1)  # Randomly selects one of the min-conflict positions.
        greedy_init[queen + 1] = min_index  # Loads the queen into her place on the next column.
        current_pos = min_index  # The row of the next queen to analyze.

    greedy_init = greedy_init.astype(int)

    return greedy_init

if __name__ == '__main__':
    # You can test your code here

    pass