import numpy as np
from constants import (
    X_TILE, O_TILE, BLANK_TILE,
    X_WINS, O_WINS, DRAW, UNFINISHED
)
from typing import Tuple, List


class Board:
    def __init__(self, size: int, reward_bad: float = -10):
        self.size = size
        self.board = np.zeros((size, size))
        self.turn = X_TILE
        self.reward_bad = reward_bad
        self.finished = False
        self.num2str = {
            X_TILE: "X",
            O_TILE: "O",
            BLANK_TILE: "-"
        }

        self.str2num = {value: key for key, value in self.num2str.items()}

    def print_board(self) -> None:
        for i in range(self.size * 2):
            print("-", end="")
        print()
        for i in range(self.size):
            print("|", end="")
            for j in range(self.size):
                print(self.num2str[self.board[i, j]], end="")
            print("|")
        for i in range(self.size * 2):
            print("-", end="")
        print()

    def get_state_from_board(self, board=None) -> str:
        if board is None:
            board = self.board.copy()
        state = ""
        for i in range(self.size):
            for j in range(self.size):
                state += self.str2num[int(board[i, j])]
        return state

    def get_board_from_state(self, state: str) -> np.ndarray:
        board_state = np.zeros((self.size, self.size))
        for i in range(self.size):
            for j in range(self.size):
                board_state[i, j] = self.num2str[state[i*self.size+j]]
        return board_state

    def reset(self) -> None:
        self.board = np.zeros((self.size, self.size))
        self.current_player = X_TILE

    def check_move(self, row: int, col: int) -> bool:
        if self.board[row, col] == BLANK_TILE:
            return True
        else:
            return False

    def get_avail_moves(self) -> List[int]:
        avail_moves = []
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i, j] == BLANK_TILE:
                    avail_moves.append(i * self.size + j)

        return avail_moves

    def check_state(self) -> int:
        # row and col
        row_sum = np.sum(self.board, axis=0)
        col_sum = np.sum(self.board, axis=1)

        if np.max(row_sum) == self.size or np.max(col_sum) == self.size:
            return X_WINS
        elif np.min(row_sum) == -self.size or np.min(col_sum) == -self.size:
            return O_WINS

        # diagonal
        main_diag = np.trace(self.board)
        sec_diag = np.trace(np.fliplr(self.board))

        if main_diag == self.size or sec_diag == self.size:
            return X_WINS
        elif main_diag == -self.size or sec_diag == -self.size:
            return O_WINS

        # check full
        product = np.prod(self.board)

        if product == 0:
            return UNFINISHED
        else:
            return DRAW

    def move(self, pos: Tuple[int, int]) -> Tuple[np.ndarray, float, bool]:
        if self.finished:
            raise ValueError("Board is already finished")
        # invalid move
        x_pos, y_pos = pos

        if 0 > x_pos or self.size <= x_pos or 0 > y_pos or self.size <= y_pos:
            raise ValueError(f"The position {pos} doesn't exist in"
                             f" board size {self.size}")

        if self.check_move(x_pos, y_pos) != 0:
            return self.board, self.reward_bad * self.turn, True

        self.board[x_pos, y_pos] = self.turn
        self.turn *= -1

        if self.check() == UNFINISHED:
            return self.board, DRAW, False
        else:
            self.finished = True
            return self.board, self.check(), False

    def compare_board(self, board: np.ndarray) -> bool:
        return np.array_equal(self.board, board)

    def compare_board_outside(
        self, board1: np.ndarray, board2: np.ndarray
    ) -> bool:
        return np.array_equal(board1, board2)

    def rotate_state_outside(self, state: str) -> str:
        board_state = self.get_board_from_state(state)
        board_rot = np.rot90(board_state)
        # print(board_state)
        # print(board_rot)
        # print("---------------------------")
        return self.get_state_from_board(board_rot)

    def flip_state_outside(self, state: str) -> str:
        board_state = self.get_board_from_state(state)
        board_flip = np.flip(board_state)
        return self.get_state_from_board(board_flip)

    def rotate_flip_state_outside(self, state: str) -> str:
        return self.flip_state_outside(self.rotate_state_outside(state))
