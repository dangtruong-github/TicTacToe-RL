import numpy as np
import copy
import pickle
import matplotlib.pyplot as plt
import os

from TicTacToeBoard import TicTacToeBoard
from BasePlayer import BasePlayer
from QLearning import RLAgentQLearning
from Game import *

board_size = 4
number_of_games = 5000

training_step_test = [100, 1000, 5000, 10000, 20000, 100000]

episodes = np.max(np.array(training_step_test))


"""## Game"""
"""## RL Agent"""

"""## Random opponent"""

"""# Main"""

rl_agent = RLAgentQLearning("QLearning".format(training_step_test[-1]), "blue", board_size)

#rl_agent.train(num_episodes=100, save_episodes=0,
               #save_step=training_step_test,
#               debug=True)

random_opp = BasePlayer("Random", "red", board_size)

game(rl_agent, random_opp, board_size, print_board=True, debug=True)

multiple_game_multiple_step_train(rl_agent, training_step_test, board_size, number_of_games)

qlearning_agent = RLAgentQLearning("QLearning".format(training_step_test[-1]), "blue", board_size)
qlearning_agent.load_policy("./Q-Learning/size={}/QLearning_epoch={}_policy.pkl".format(board_size, 20000))

sarsa_agent = RLAgentQLearning("SARSA".format(training_step_test[-1]), "green", board_size)
sarsa_agent.load_policy("./SARSA/size={}/SARSA_epoch={}_policy.pkl".format(board_size, 20000))

qlearning_wins = 0

sarsa_wins = 0

draw = 0

for i in range(5000):
    _, res1 = game(qlearning_agent, sarsa_agent, board_size, print_board=True, debug=True)
    _, res2 = game(sarsa_agent, qlearning_agent, board_size, print_board=True, debug=True)

    qlearning_wins += 1 if res1 == X_WIN else 0
    qlearning_wins += 1 if res2 == O_WIN else 0

    sarsa_wins += 1 if res1 == O_WIN else 0
    sarsa_wins += 1 if res2 == X_WIN else 0

    draw += 1 if res1 == DRAW else 0
    draw += 1 if res2 == DRAW else 0

print(f"QLearning wins: {qlearning_wins}")
print(f"SARSA wins: {sarsa_wins}")
print(f"Draw: {draw}")