import numpy as np

from constants import *

class TicTacToeBoard:
    def __init__(self, board_size):
        self.size = board_size
        self.board = np.zeros((board_size, board_size))
        self.current_player = X_TILE
        self.state_representation = {
            "X": X_TILE,
            "O": O_TILE,
            "-": EMPTY_TILE
        }

        self.state_representation_reverse = {
            X_TILE: "X",
            O_TILE: "O",
            EMPTY_TILE: "-"
        }

    def get_state(self):
        state = ""
        for i in range(self.size):
            for j in range(self.size):
                state += self.state_representation_reverse[int(self.board[i, j])]
        return state

    def get_state_from_board(self, board):
        state = ""
        for i in range(self.size):
            for j in range(self.size):
                state += self.state_representation_reverse[int(board[i, j])]
        return state

    def get_board_from_state(self, state):
        board_state = np.zeros((self.size, self.size))
        for i in range(self.size):
            for j in range(self.size):
                board_state[i, j] = self.state_representation[state[i * self.size + j]]
        return board_state

    def print_board(self):
        print("--------------------")
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i, j] == X_TILE:
                    print("X", end=" ")
                elif self.board[i, j] == O_TILE:
                    print("O", end=" ")
                else:
                    print("-", end=" ")
            print()
        print("--------------------")

    def reset(self):
        self.board = np.zeros((self.size, self.size))
        self.current_player = X_TILE

    def check_move(self, row, col):
        if self.board[row, col] == EMPTY_TILE:
            return True
        else:
            return False

    def get_avail_moves(self):
        avail_moves = []
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i, j] == EMPTY_TILE:
                    avail_moves.append(i * self.size + j)

        return avail_moves

    def make_move(self, row, col, test=False):
        if self.check_move(row, col) is False:
            return {
                "state": "Invalid move",
                "reward": -10 * self.current_player,
                "end": False
            }

        self.board[row, col] = self.current_player
        self.current_player *= -1

        state = self.get_state()
        reward = self.check_state()
        end = True if reward != UNFINISHED else False
        if reward == UNFINISHED:
            reward = 0

        if test:
            # revert back
            self.board[row, col] = EMPTY_TILE
            self.current_player *= -1

        return {
            "state": state,
            "reward": reward,
            "end": end
        }

    def compare_state(self, state):
        return np.array_equal(self.board, state)

    def compare_state_outside(self, state1, state2):
        return np.array_equal(state1, state2)

    def rotate_state_outside(self, state):
        board_state = self.get_board_from_state(state)
        board_rot = np.rot90(board_state)
        # print(board_state)
        # print(board_rot)
        # print("---------------------------")
        return self.get_state_from_board(board_rot)

    def flip_state_outside(self, state):
        board_state = self.get_board_from_state(state)
        board_flip = np.flip(board_state)
        return self.get_state_from_board(board_flip)

    def rotate_flip_state_outside(self, state):
        return self.flip_state_outside(self.rotate_state_outside(state))

    def check_state(self):
        # Check rows
        for i in range(self.size):
            if np.all(self.board[i, :] == X_TILE):
                return X_WIN
            elif np.all(self.board[i, :] == O_TILE):
                return O_WIN

        # Check columns
        for j in range(self.size):
            if np.all(self.board[:, j] == X_WIN):
                return X_WIN
            elif np.all(self.board[:, j] == O_WIN):
                return O_WIN

        # Check diagonals
        if np.all(np.diag(self.board) == X_WIN) or np.all(np.diag(np.fliplr(self.board)) == X_WIN):
            return X_WIN
        elif np.all(np.diag(self.board) == O_WIN) or np.all(np.diag(np.fliplr(self.board)) == O_WIN):
            return O_WIN

        # Check for draw
        if np.all(self.board != EMPTY_TILE):
            return DRAW

        # Game is unfinished
        return UNFINISHED
