from pandas import *
from pprint import pprint
import numpy as np
from doe import *
from tac import *
import math
import time
import random

board = np.zeros((3, 3), dtype=np.int8)
magic_number = 9


def CheckCat(board):
    # print(board)
    return not (board == 0).any()


def CheckVictory(board, x, y):  # From https://codereview.stackexchange.com/questions/24764/tic-tac-toe-victory-check?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    # check if previous move caused a win on vertical line
    if board[0][y] == board[1][y] == board[2][y]:
        return True
    # check if previous move caused a win on horizontal line
    if board[x][0] == board[x][1] == board[x][2]:
        return True
    # check if previous move was on the main diagonal and caused a win
    if x == y and board[0][0] == board[1][1] == board[2][2]:
        return True
    # check if previous move was on the secondary diagonal and caused a win
    if x + y == 2 and board[0][2] == board[1][1] == board[2][0]:
        return True
    return False


if __name__ == "__main__":
    print(DataFrame(board.transpose()))
    state_size = np.ravel(board).shape
    agentOne = Doe(state_size)
    agentTwo = Tac(2)
    EPISODES = 1000000
    BATCH_SIZE = 32
    PRINT_RATE = 250
    recent_wins = 0
    recent_losses = 0
    recent_cats = 0
    for e in range(EPISODES):
        while True:
            board, row_action, column_action = agentOne.fill_board(
                agentOne.act(board), board, 1)
            if CheckVictory(board, row_action, column_action):
                agentOne.remember_game(1)
                board = np.zeros((3, 3), dtype=np.int8)
                recent_wins += 1
                break
            if CheckCat(board):
                agentOne.remember_game(0)
                board = np.zeros((3, 3), dtype=np.int8)
                recent_cats += 1
                break
            board, row_action, column_action = agentTwo.act(board)
            if CheckVictory(board, row_action, column_action):
                agentOne.remember_game(-1)
                board = np.zeros((3, 3), dtype=np.int8)
                recent_losses += 1
                break
            if CheckCat(board):
                agentOne.remember_game(0)
                board = np.zeros((3, 3), dtype=np.int8)
                recent_cats += 1
                break
        if (e+1) % BATCH_SIZE == 0:
            agentOne.train()
        if (e+1) % PRINT_RATE == 0:
            print("Win %: {:.2}, Loss %: {:.2}, Cat %: {:.2}, E: {:.2}".format(recent_wins/PRINT_RATE,
                                                                               recent_losses/PRINT_RATE, recent_cats/PRINT_RATE, agentOne.exploration))
            recent_wins = 0
            recent_losses = 0
            recent_cats = 0
