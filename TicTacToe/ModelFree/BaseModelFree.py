import pickle
import os
import numpy as np

from abc import abstractmethod

from BaseAgent import BaseAgent
from constants import (
    X_TILE
)


class BaseModelFree(BaseAgent):
    def __init__(
        self,
        size: int,
        lr: float = 1e-3,
        epsilon: float = 0.1,
        gamma: float = 0.99,
        name: str = "ModelFree",
        path_save: str = "./saved_models/model_free",
    ):
        super(BaseModelFree, self).__init__(
            size=size, lr=lr, epsilon=epsilon,
            gamma=gamma, name=name
        )
        self.path_save = path_save
        self.q_table = {}

    def choose_action(self, board, debug=False, infer=False):
        # Choose A from S using policy derived from Q (e.g., epsilon-greedy)
        random_number = np.random.uniform(0, 1)
        epsilon_now = 0.1 if infer else self.current_epsilon
        rotation_id = 0

        if random_number < epsilon_now:
            # Explore: choose a random action
            action = np.random.choice(board.get_avail_moves())
        else:
            # Greedy
            state = board.get_state()
            in_table, (state_rotated, rotation_id) = self.check_in_table(state)
            if not in_table:
                self.init_value_strategy(state_rotated, board.current_player)

            if board.current_player == X_TILE:
                action = np.argmax(self.q_table[state_rotated])
            else:
                action = np.argmin(self.q_table[state_rotated])

            if debug:
                print(f"Inside action: {state_rotated}")
                print(f"board.current_player: {board.current_player}")
                print(f"Inside action: {action}")

        action = (action // board.size, action % board.size)

        if True:
            if rotation_id == 1:
                action = (board.size - 1 - action[1], action[0])
            elif rotation_id == 2:
                action = (
                    board.size - 1 - action[0],
                    board.size - 1 - action[1]
                )
            elif rotation_id == 3:
                action = (action[1], board.size - 1 - action[0])

        return action, rotation_id

    def load_model(self, path_load):
        with open(path_load, "rb") as f:
            self.q_table = pickle.load(f)

    def save_model(self, path_save=None):
        if path_save is None:
            path_save = os.path.join
            path_save = "{}/size={}/{}_policy.pkl".format(
                self.path_save, self.size, self.name
            )

    def make_move(self, board, debug=False):
        assert board.size == self.size, "Board size does not match"

        action, rotation_id = self.choose_action(board)

        if debug:
            print(f"Agent {board.turn} action: {action}")
            print(f"Rotation id: {rotation_id}")

        return action

    # abstract methods
    @abstractmethod
    def train(self):
        pass
