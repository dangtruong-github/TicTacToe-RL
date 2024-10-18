import numpy as np
import pickle
import os
import copy

from TrainStrategy.BasePlayer import BasePlayer
from TicTacToeBoard import TicTacToeBoard
from constants import *

class RLAgentQLearning(BasePlayer):
    def __init__(self, name, color, board_size, gamma=1, alpha=0.1, epsilon=1):
        super(RLAgentQLearning, self).__init__(name, color, board_size)

        # Algorithm parameters: step size alpha: (0, 1], small epsilon > 0
        self.gamma = gamma
        self.alpha = alpha
        self.init_epsilon = epsilon

        # Initialize Q(s, a) arbitrarily except that Q(terminal, ·) = 0
        self.q_table = {}
        self.pi = {}

        self.current_epsilon = 0


    def epsilon_scheduler(self, num_episodes):
        # from 1 to 0.1, linear
        self.epsilon = np.full(num_episodes, 0.1)
        self.current_epsilon = self.epsilon[0]

    def load_policy(self, path_load):
        with open(path_load, "rb") as f:
            self.q_table = pickle.load(f)

    def save_policy(self, path_save=None):
        if path_save is None:
            path_save = "./SARSA/size={}/{}_policy.pkl".format(self.board_size, self.name)

        # get folder
        folder_to_make = path_save.split("/")[:-1]
        os.makedirs("/".join(folder_to_make), exist_ok=True)

        with open(path_save, "wb") as f:
            pickle.dump(self.q_table, f)

    def init_value_strategy(self, state, turn):
        """
        if turn == X_TILE:
            self.q_table[state] = np.random.uniform(-0.2, 0, (self.board_size, self.board_size))
            # Replace the diagonal elements with values from the uniform distribution [0, 0.2]
            np.fill_diagonal(self.q_table[state], np.random.uniform(0, 0.2, self.board_size))
        else:
            self.q_table[state] = np.random.uniform(0, 0.2, (self.board_size, self.board_size))
            # Replace the diagonal elements with values from the uniform distribution [0, 0.2]
            np.fill_diagonal(self.q_table[state], np.random.uniform(-0.2, 0, self.board_size))

        # Flatten
        self.q_table[state] = self.q_table[state].flatten()
        """

        # Initialize Q(s, a) arbitrarily except that Q(terminal, ·) = 0
        self.q_table[state] = np.random.uniform(-0.1, 0.1, size=(self.board_size ** 2))

    def check_in_table(self, state):
        orig_state = copy.deepcopy(state)
        # rotate
        for i in range(4):
            if state in self.q_table:
                return True, (state, i)

            board = TicTacToeBoard(self.board_size)
            state = board.rotate_state_outside(state)

        assert orig_state == state, "State rotation failed"

        return False, (state, 0)

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
                action = (board.size - 1 - action[0], board.size - 1 - action[1])
            elif rotation_id == 3:
                action = (action[1], board.size - 1 - action[0])

        return action, rotation_id

    def train(self, num_episodes, save_episodes=0, save_step=[], debug=False):
        if save_episodes > num_episodes or save_episodes == 0:
            save_episodes = (num_episodes // 10)

        self.epsilon_scheduler(num_episodes)

        # Loop for each episode:
        for episode in range(num_episodes):
            # Initialize S
            board = TicTacToeBoard(self.board_size)
            board.reset()

            self.current_epsilon = self.epsilon[episode]

            # Loop for each step of episode:
            while board.check_state() == UNFINISHED:
                # Initialize Q(s, a) arbitrarily except that Q(terminal, ·) = 0
                state = board.get_state()
                in_table, (state_rotated, rotation_id) = self.check_in_table(state)
                if in_table == False:
                    self.init_value_strategy(state_rotated, board.current_player)

                # Choose A from S using policy derived from Q (e.g., epsilon-greedy)
                action, _ = self.choose_action(board, debug=debug)

                if rotation_id == 1:
                    rotated_action = (action[1], board.size - 1 - action[0])
                elif rotation_id == 2:
                    rotated_action = (board.size - 1 - action[0], board.size - 1 - action[1])
                elif rotation_id == 3:
                    rotated_action = (board.size - 1 - action[1], action[0])
                else:
                    rotated_action = action

                # Take action A, observe R, S0
                move_return = board.make_move(action[0], action[1], test=True)

                next_state = move_return["state"]
                reward = move_return["reward"]
                end = move_return["end"]

                # Update
                # Invalid
                if next_state == "Invalid move":
                    try:
                        if rotation_id != 0:
                            board.print_board()
                            print(f"Action: {action}")
                            print(f"Rotated action: {rotated_action}")
                            print("Rotated board:")
                            for i in range(self.board_size):
                                for j in range(self.board_size):
                                    print(state_rotated[i * self.board_size + j], end=" ")
                                print()
                            print(self.q_table[state_rotated].reshape(self.board_size, self.board_size))
                            print(self.q_table[state_rotated][rotated_action[0] * self.board_size + rotated_action[1]])
                            print(reward)
                        if self.q_table[state_rotated][rotated_action[0] * self.board_size + rotated_action[1]] == reward:
                            board.print_board()
                            print(f"Action: {action}")
                            print(f"Rotated action: {rotated_action}")
                            print("Rotated board:")
                            for i in range(self.board_size):
                                for j in range(self.board_size):
                                    print(state_rotated[i * self.board_size + j], end=" ")
                                print()
                            print(self.q_table[state_rotated].reshape(self.board_size, self.board_size))
                            print(self.q_table[state_rotated][rotated_action[0] * self.board_size + rotated_action[1]])
                            print(reward)
                            raise Exception("Invalid update")
                    except:
                        assert (self.q_table[state_rotated][rotated_action[0] * self.board_size + rotated_action[1]] == reward) == False
                    self.q_table[state_rotated][rotated_action[0] * self.board_size + rotated_action[1]] = reward
                    continue
                if rotation_id != 0:
                    board.print_board()
                    print(f"Action: {action}")
                    print(f"Rotated action: {rotated_action}")
                    print("Rotated board:")
                    for i in range(self.board_size):
                        for j in range(self.board_size):
                            print(state_rotated[i * self.board_size + j], end=" ")
                        print()
                    print(self.q_table[state_rotated].reshape(self.board_size, self.board_size))
                    print(self.q_table[state_rotated][rotated_action[0] * self.board_size + rotated_action[1]])
                    print(reward)
                if self.q_table[state_rotated][rotated_action[0] * self.board_size + rotated_action[1]] == 10 or self.q_table[state_rotated][rotated_action[0] * self.board_size + rotated_action[1]] == -10:
                    board.print_board()
                    print(f"Action: {action}")
                    print(f"Rotated action: {rotated_action}")
                    print("Rotated board:")
                    for i in range(self.board_size):
                        for j in range(self.board_size):
                            print(state_rotated[i * self.board_size + j], end=" ")
                        print()
                    print(self.q_table[state_rotated].reshape(self.board_size, self.board_size))
                    print(self.q_table[state_rotated][rotated_action[0] * self.board_size + rotated_action[1]])
                    print(reward)
                    raise Exception("Invalid update")

                if end == True:
                    self.q_table[state_rotated][rotated_action[0] * self.board_size + rotated_action[1]] = reward
                    break

                next_in_table, (next_state_rotated, _) = self.check_in_table(next_state)
                if next_in_table == False:
                    self.init_value_strategy(next_state_rotated, board.current_player * -1)

                if board.current_player == X_TILE:
                    q_next = np.min(self.q_table[next_state_rotated])
                else:
                    q_next = np.max(self.q_table[next_state_rotated])

                td = reward + self.gamma * q_next - self.q_table[state_rotated][rotated_action[0] * self.board_size + rotated_action[1]]
                self.q_table[state_rotated][rotated_action[0] * self.board_size + rotated_action[1]] += self.alpha * td

                # Update S
                board.make_move(action[0], action[1])

            if (episode + 1) % save_episodes == 0:
                self.save_policy()
                print(f"Episode {episode} saved")

            if (episode + 1) in save_step:
                path_save = "./Q-Learning/size={}/{}_epoch={}_policy.pkl".format(self.board_size, self.name, episode+1)
                self.save_policy(path_save=path_save)
                print(f"Step {episode} saved")


        self.save_policy()
        print(f"Episode total {num_episodes} saved")

    def act(self, board, debug=False):
        assert board.size == self.board_size, "Board size does not match"

        action, rotation_id = self.choose_action(board)

        if debug:
            print(f"Agent {board.current_player} action: {action}")
            print(f"Rotation id: {rotation_id}")

        return action, 0
