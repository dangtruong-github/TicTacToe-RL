import numpy as np
import matplotlib.pyplot as plt

from TicTacToeBoard import TicTacToeBoard
from constants import *
from TrainStrategy.BasePlayer import BasePlayer
from TrainStrategy.QLearning import RLAgentQLearning

def plot_win_rate(agent_1_wins, agent_2_wins, draw, agent_1, agent_2, board_size, save_fig_path=None):
    values = np.array([agent_1_wins, agent_2_wins, draw])
    percentages = values / np.sum(values) * 100

    labels = [
        "{} wins".format(agent_1.name),
        "{} wins".format(agent_2.name),
        "Draw"
    ]

    colors = [agent_1.color, agent_2.color, "grey"]

    plt.pie(percentages, labels=labels, autopct='%1.1f%%', colors=colors)
    plt.title("Win loss stats {}x{}".format(board_size, board_size))
    if save_fig_path is not None:
        plt.savefig(save_fig_path)
    plt.show()

def game(agent_1, agent_2, board_size, print_board=True, debug=False):
    board = TicTacToeBoard(board_size)
    board.reset()

    moves = []
    states_history = []
    agent_1_rotation = []
    agent_2_rotation = []

    turn = X_TILE

    while True:
        if turn == X_TILE:
            action, rotation = agent_1.act(board, debug=debug)
            agent_1_rotation.append(rotation)
        else:
            action, rotation = agent_2.act(board, debug=debug)
            agent_2_rotation.append(rotation)

        if debug:
            print(action)

        result = board.make_move(action[0], action[1])
        states_history.append(board.get_state())
        moves.append(action)
        turn *= -1

        if print_board:
            board.print_board()

        if result["end"]:
            return (moves, states_history, agent_1_rotation, agent_2_rotation), result["reward"]

def multiple_games(agent_1, agent_2, board_size, num_games, debug=False, save_fig_path=None):
    agent_1_wins = 0
    agent_2_wins = 0
    draw = 0

    for i in range(num_games):
        history_X, resultX = game(agent_1, agent_2, board_size, print_board=False, debug=debug)
        history_O, resultO = game(agent_2, agent_1, board_size, print_board=False, debug=debug)

        moves_X, states_X, rotation_X, _ = history_X
        moves_O, states_O, _, rotation_O = history_O

        agent_1_wins += 1 if resultX == X_WIN else 0
        agent_1_wins += 1 if resultO == O_WIN else 0

        agent_2_wins += 1 if resultX == O_WIN else 0
        agent_2_wins += 1 if resultO == X_WIN else 0

        draw += 1 if resultX == DRAW else 0
        draw += 1 if resultO == DRAW else 0

        if debug:
            if resultX == O_WIN:
                print(f"Move X: {states_X}")
            if resultO == X_WIN:
                print(f"Move O: {states_O}")

    plot_win_rate(agent_1_wins, agent_2_wins, draw, agent_1, agent_2, board_size, save_fig_path=save_fig_path)

    return agent_1_wins, agent_2_wins, draw

def multiple_game_multiple_step_train(rl_agent, train_step_test, board_size, num_games):
    win_rates = []
    lose_rates = []
    draw_rates = []

    random_opp = BasePlayer("Random", "red", board_size)

    for step_eval in train_step_test:
        rl_agent_test = RLAgentQLearning("QLearning{}".format(step_eval), "blue", board_size)
        rl_agent_test.load_policy("./Q-Learning/size={}/QLearning_epoch={}_policy.pkl".format(board_size, step_eval))

        agent_wins, random_wins, draw = multiple_games(rl_agent_test, random_opp, board_size, num_games, debug=False)

        pct_wins, pct_lose, pct_draw = agent_wins / (2 * num_games), random_wins / (2 * num_games), draw / (2 * num_games)

        print(f"Agent wins: {agent_wins}")
        print(f"Random wins: {random_wins}")
        print(f"Draw: {draw}")

        win_rates.append(pct_wins)
        lose_rates.append(pct_lose)
        draw_rates.append(pct_draw)

    plt.plot(train_step_test, win_rates, color=rl_agent.color)
    plt.plot(train_step_test, lose_rates, color=random_opp.color)
    plt.plot(train_step_test, draw_rates, color="grey")
    plt.xlabel("Training step")
    plt.ylabel("Win rate")
    plt.legend([rl_agent.name, "Random wins", "Draw"])
    plt.savefig("./Q-Learning/size={}/win_rate_by_train_step.png".format(board_size))
    plt.show()

