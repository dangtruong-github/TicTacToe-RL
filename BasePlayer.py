import numpy as np

class BasePlayer:
    def __init__(self, name, color, board_size):
        self.board_size = board_size
        self.color = color
        self.name = name

    def act(self, board, debug=False):
        assert board.size == self.board_size, "Board size does not match"

        # random move
        avail_moves = board.get_avail_moves()
        move = np.random.choice(avail_moves)
        move = (move // board.size, move % board.size)

        return move, 0
