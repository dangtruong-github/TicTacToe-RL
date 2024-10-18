import numpy as np
import yaml

from TrainStrategy.BasePlayer import BasePlayer
from TrainStrategy.QLearning import RLAgentQLearning
from Game import *
from utils import *

with open("config.yml", 'r') as f:
    dict_config = yaml.load(f, Loader=yaml.FullLoader)

args = dict2namespace(dict_config)
args = args.main_game

print(dict_config)
print(args)


episodes = np.max(np.array(args.training_step_test))


rl_agent = RLAgentQLearning("QLearning".format(args.training_step_test[-1]), "blue", args.board_size)


random_opp = BasePlayer("Random", "red", args.board_size)

game(rl_agent, random_opp, args.board_size, print_board=True, debug=True)

multiple_game_multiple_step_train(
    rl_agent,
    args.training_step_test,
    args.board_size,
    args.number_of_games)

qlearning_agent = RLAgentQLearning(
    "QLearning".format(args.training_step_test[-1]),
    "blue", args.board_size)
qlearning_agent.load_policy("./Q-Learning/size={}/QLearning_epoch={}_policy.pkl".format(args.board_size, 20000))

sarsa_agent = RLAgentQLearning("SARSA".format(args.training_step_test[-1]), "green", args.board_size)
sarsa_agent.load_policy("./SARSA/size={}/SARSA_epoch={}_policy.pkl".format(args.board_size, 20000))

qlearning_wins = 0

sarsa_wins = 0

draw = 0

for i in range(5000):
    _, res1 = game(qlearning_agent, sarsa_agent, args.board_size, print_board=True, debug=True)
    _, res2 = game(sarsa_agent, qlearning_agent, args.board_size, print_board=True, debug=True)

    qlearning_wins += 1 if res1 == X_WIN else 0
    qlearning_wins += 1 if res2 == O_WIN else 0

    sarsa_wins += 1 if res1 == O_WIN else 0
    sarsa_wins += 1 if res2 == X_WIN else 0

    draw += 1 if res1 == DRAW else 0
    draw += 1 if res2 == DRAW else 0

print(f"QLearning wins: {qlearning_wins}")
print(f"SARSA wins: {sarsa_wins}")
print(f"Draw: {draw}")