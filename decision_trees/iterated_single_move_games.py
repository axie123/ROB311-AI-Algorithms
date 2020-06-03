from abc import ABC, abstractmethod
import numpy as np


class SingleMoveGamePlayer(ABC):
    """
    Abstract base class for a symmetric, zero-sum single move game player.
    """
    def __init__(self, game_matrix: np.ndarray):
        self.game_matrix = game_matrix
        self.n_moves = game_matrix.shape[0]
        super().__init__()

    @abstractmethod
    def make_move(self) -> int:
        pass


class IteratedGamePlayer(SingleMoveGamePlayer):
    """
    Abstract base class for a player of an iterated symmetric, zero-sum single move game.
    """
    def __init__(self, game_matrix: np.ndarray):
        super(IteratedGamePlayer, self).__init__(game_matrix)

    @abstractmethod
    def make_move(self) -> int:
        pass

    @abstractmethod
    def update_results(self, my_move, other_move):
        """
        This method is called after each round is played
        :param my_move: the move this agent played in the round that just finished
        :param other_move:
        :return:
        """

        pass

    @abstractmethod
    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.)
        :return:
        """
        pass


class UniformPlayer(IteratedGamePlayer):
    def __init__(self, game_matrix: np.ndarray):
        super(UniformPlayer, self).__init__(game_matrix)

    def make_move(self) -> int:
        """

        :return:
        """
        return np.random.randint(0, self.n_moves)

    def update_results(self, my_move, other_move):
        """
        The UniformPlayer player does not use prior rounds' results during iterated games.
        :param my_move:
        :param other_move:
        :return:
        """
        pass

    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.)
        :return:
        """
        pass


class FirstMovePlayer(IteratedGamePlayer):
    def __init__(self, game_matrix: np.ndarray):
        super(FirstMovePlayer, self).__init__(game_matrix)

    def make_move(self) -> int:
        """
        Always chooses the first move
        :return:
        """
        return 0

    def update_results(self, my_move, other_move):
        """
        The FirstMovePlayer player does not use prior rounds' results during iterated games.
        :param my_move:
        :param other_move:
        :return:
        """
        pass

    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.)
        :return:
        """
        pass


class CopycatPlayer(IteratedGamePlayer):
    def __init__(self, game_matrix: np.ndarray):
        super(CopycatPlayer, self).__init__(game_matrix)
        self.last_move = np.random.randint(self.n_moves)

    def make_move(self) -> int:
        """
        Always copies the last move played
        :return:
        """
        return self.last_move

    def update_results(self, my_move, other_move):
        """
        The CopyCat player simply remembers the opponent's last move.
        :param my_move:
        :param other_move:
        :return:
        """
        self.last_move = other_move

    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.)
        :return:
        """
        self.last_move = np.random.randint(self.n_moves)


def play_game(player1, player2, game_matrix: np.ndarray, N: int = 1000) -> (int, int):
    """

    :param player1: instance of an IteratedGamePlayer subclass for player 1
    :param player2: instance of an IteratedGamePlayer subclass for player 2
    :param game_matrix: square payoff matrix
    :param N: number of rounds of the game to be played
    :return: tuple containing player1's score and player2's score
    """
    p1_score = 0.0
    p2_score = 0.0
    n_moves = game_matrix.shape[0]
    legal_moves = set(range(n_moves))
    for idx in range(N):
        move1 = player1.make_move()
        move2 = player2.make_move()
        if move1 not in legal_moves:
            print("WARNING: Player1 made an illegal move: {:}".format(move1))
            if move2 not in legal_moves:
                print("WARNING: Player2 made an illegal move: {:}".format(move2))
            else:
                p2_score += np.max(game_matrix)
                p1_score -= np.max(game_matrix)
            continue
        elif move2 not in legal_moves:
            print("WARNING: Player2 made an illegal move: {:}".format(move2))
            p1_score += np.max(game_matrix)
            p2_score -= np.max(game_matrix)
            continue
        player1.update_results(move1, move2)
        player2.update_results(move2, move1)

        p1_score += game_matrix[move1, move2]
        p2_score += game_matrix[move2, move1]

    return p1_score, p2_score


class StudentAgent(IteratedGamePlayer):
    """
    I worked on this together with a group of friends, so we used very similar strategy.

    Strategy involves using the last 2 moves of the opponent and produces the most optimal move
    as the 3rd move in the sequence based on probabilistic reasoning. Uses a softmax probability and
    a uniform probability guess to determine the most optimal move for the player. The initialization
    starts with setting up the matrix of the state of the game, and the recent and past plays of the opponent
    and player. A dictionary is used to keep track of all the frequencies of possible 3-move sequences using
    0, 1, and 2. The copycat and the first-move player are detected by checking the last 4 moves and
    checking if only one move has been made by the opponent respectively. A list of weight values for all
    possible moves would then be generated for the player. The probability of each given move is generated by
    the softmax function. Each softmax probability is compared with a uniform probability guess, and the optimal
    action would be assigned accordingly based on which softmax probability is the closest to the uniform guess.
    The player's and opponent's actions, and the freq dictionary would then be updated for the next cycle.
    """
    def __init__(self, game_matrix: np.ndarray):
        """
        Initialize your game playing agent. here
        :param game_matrix: square payoff matrix for the game being played.
        """
        super(StudentAgent, self).__init__(game_matrix)
        self.game = game_matrix  # The matrix of the state of the game.
        self.opp_play = '00'  # Stores the current/most recent play of the opponent.
        self.past_plays = []  # Stores all the past plays of the opponent.
        self.player_moves = []  # Stores all the past plays of the player.
        self.rock_paper_scissors = {'000': 3, '001': 3, '002': 3,  # Keeps track of the frequencies of 3-move sequences.
                                    '010': 3, '011': 3, '012': 3,  # 0 - rock, 1 - paper, 2 - scissors.
                                    '020': 3, '021': 3, '022': 3,
                                    '100': 3, '101': 3, '102': 3,
                                    '110': 3, '111': 3, '112': 3,
                                    '120': 3, '121': 3, '122': 3,
                                    '200': 3, '201': 3, '202': 3,
                                    '210': 3, '211': 3, '212': 3,
                                    '220': 3, '221': 3, '222': 3}

    def make_move(self) -> int:
        """
        Play your move based on previous moves or whatever reasoning you want.
        :return: an int in (0, ..., n_moves-1) representing your move
        """
        move = 0

        if len(self.player_moves) >= 5:  # Checking for the copycat:
            if self.player_moves[-5:-1] == self.past_plays[-4:]:  # If the past 4 actions of the opponent matches that of the player: return.
                return (self.player_moves[-1] + 1) % 3

        if len(self.past_plays) > 0 and len(set(self.past_plays)) < 2:  # Checks for the first move player:
            return (self.past_plays[0] + 1) % 3

        # Gives the proportional weight values of each possible move.
        pos_move = [self.rock_paper_scissors[self.opp_play + '0'], self.rock_paper_scissors[self.opp_play + '1'], self.rock_paper_scissors[self.opp_play + '2']]
        pos_move_prob = np.exp(pos_move) / np.sum(np.exp(pos_move)) # Calculates the softmax probability of each move.
        opt_move_prob = np.random.uniform()  # Gives a guess for the probability of the optimal move.
        if pos_move_prob[0] > opt_move_prob:  # If statements determine the move that corresponds the most with the guess.
            return 1  # Gives the move corresponding to 1.
        elif pos_move_prob[0] + pos_move_prob[1] > opt_move_prob:
            return 2  # Gives the move corresponding to 2.
        else:
            return 0  # Gives the move corresponding to 0.

    def update_results(self, my_move, opp_move):
        """
        Update your agent based on the round that was just played.
        :param my_move:
        :param other_move:
        :return: nothing
        """
        self.opp_play += str(opp_move)  # Stores the opponent's last move.
        self.rock_paper_scissors[self.opp_play] += 1  # Updates the frequency of the corresponding move.
        self.opp_play = self.opp_play[1:]  # Shifts for the most recent opponent moves.
        self.past_plays.append(opp_move)  # Uploads the recent move onto a list of all of the opponent's moves.
        self.player_moves.append(my_move)  # Uploads the recent move onto a list of all of the player's moves.

    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.).
        :return: nothing
        """
        self.opp_play = '00'
        self.past_plays = []
        self.player_moves = []
        self.rock_paper_scissors = {'000': 3, '001': 3, '002': 3,
                                    '010': 3, '011': 3, '012': 3,
                                    '020': 3, '021': 3, '022': 3,
                                    '100': 3, '101': 3, '102': 3,
                                    '110': 3, '111': 3, '112': 3,
                                    '120': 3, '121': 3, '122': 3,
                                    '200': 3, '201': 3, '202': 3,
                                    '210': 3, '211': 3, '212': 3,
                                    '220': 3, '221': 3, '222': 3}


if __name__ == '__main__':
    """
    Simple test on standard rock-paper-scissors
    The game matrix's row (first index) is indexed by player 1 (P1)'s move (i.e., your move)
    The game matrix's column (second index) is indexed by player 2 (P2)'s move (i.e., the opponent's move)
    Thus, for example, game_matrix[0, 1] represents the score for P1 when P1 plays rock and P2 plays paper: -1.0 
    because rock loses to paper.
    """
    game_matrix = np.array([[0.0, -1.0, 1.0],
                            [1.0, 0.0, -1.0],
                            [-1.0, 1.0, 0.0]])
    uniform_player = UniformPlayer(game_matrix)
    first_move_player = FirstMovePlayer(game_matrix)
    uniform_score, first_move_score = play_game(uniform_player, first_move_player, game_matrix)

    print("Uniform player's score: {:}".format(uniform_score))
    print("First-move player's score: {:}".format(first_move_score))

    # Now try your agent
    student_player = StudentAgent(game_matrix)
    student_score, first_move_score = play_game(student_player, first_move_player, game_matrix)

    print("Your player's score: {:}".format(student_score))
    print("First-move player's score: {:}".format(first_move_score))
