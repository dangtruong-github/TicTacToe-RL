import numpy as np

from .BaseModelFree import BaseModelFree
from Board import Board
from constants import (
    X_TILE, O_TILE,
    UNFINISHED
)


class DPAgent(BaseModelFree):
    def __init__(self, size):
        super(DPAgent, self).__init__(size)
        self.total_states_visited = 0

    def train(self):
        board = Board(self.size)
        board.reset()
        # Agent is X
        self.get_max_state(board)

        # Agent is O, goes later
        for i in range(self.size * self.size):
            move = (i // self.size, i % self.size)
            state = board.move(move[0], move[1], test=True)["state"]
            new_board_state = board.get_board_from_state(state)
            new_board = Board(self.size)
            new_board.board = new_board_state
            new_board.current_player = board.current_player * -1
            print(state)
        print("Train success")
        print(f"Total states visited: {self.total_states_visited}")

    def get_max_state(self, board):
        if board.get_state() in self.q_table.keys():
            return

        if board.check_state() != UNFINISHED:
            self.q_table[board.get_state()] = np.full(
                shape=(self.size, self.size),
                fill_value=board.check_state()
            )
            return

        self.q_table[board.get_state()] = np.full(
            shape=(self.size, self.size),
            fill_value=-10 * board.current_player
        )

        # print(board.get_state())
        self.total_states_visited += 1
        avail_moves = board.get_avail_moves()

        # count X in new_board_state
        sum_total = np.sum(board.board)
        player_cur = X_TILE if sum_total == 0 else O_TILE

        if player_cur == X_TILE:
            assert sum_total == 0, "Sum invalid for X"
        else:
            assert sum_total == 1, "Sum invalid for O"

        assert board.turn == player_cur, "Current player does not match"

        for move in avail_moves:
            move = (move // board.size, move % board.size)
            state = board.move(move[0], move[1], test=True)["state"]

            # print(state)
            # print(type(state))
            # print("**************")
            state_rotate = board.rotate_state_outside(state)
            state_flip = board.flip_state_outside(state)
            state_rotate_flip = board.rotate_flip_state_outside(state)

            if (state not in self.q_table) and (state_rotate not in self.q_table) and (state_flip not in self.q_table) and (state_rotate_flip not in self.q_table):
                # print(state)
                # print(self.q_table[state])
                new_board_state = board.get_board_from_state(state)
                new_board = Board(self.size)
                new_board.board = new_board_state
                new_board.current_player = board.current_player * -1

                # count X in new_board_state
                sum_total = np.sum(new_board_state)
                player_cur = X_TILE if sum_total == 0 else O_TILE

                if player_cur == X_TILE:
                    assert sum_total == 0, "Sum invalid for X"
                else:
                    assert sum_total == 1, "Sum invalid for O"

                assert new_board.turn == player_cur, "Current player does not match"

                self.get_max_state(new_board)

            if state not in self.q_table:
                if state_rotate in self.q_table:
                    state = state_rotate
                elif state_flip in self.q_table:
                    state = state_flip
                elif state_rotate_flip in self.q_table:
                    state = state_rotate_flip

            # print(self.q_table[board.get_state()])

            if board.current_player == X_TILE:
                self.q_table[board.get_state()][move[0]][move[1]] = self.gamma * np.min(
                  self.q_table[state]
                )
            else:
                self.q_table[board.get_state()][move[0]][move[1]] = self.gamma * np.max(
                  self.q_table[state]
                )

        if self.total_states_visited % self.save_states == 0:
            print(f"Finish visiting {self.total_states_visited} states")
