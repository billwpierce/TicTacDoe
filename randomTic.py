from pandas import *
from pprint import pprint
import numpy as np
from doe import *
from tac import *
import math
import time
import random
from enum import Enum

board = np.zeros((3, 3), dtype=np.int8)
magic_number = 9


def personAct(board):
    print(DataFrame(board))
    copy = board
    while(True):
        try:
            location = [int(n) for n in (
                input("Enter a movement location, in the form '0,1': ").split(','))]
            if len(location) != 2:
                print("Invalid Input: incorrect input shape.")
            elif location[0] < 0 or location[0] > 2:
                print("Invalid Input: first index out of range.")
            elif location[1] < 0 or location[1] > 2:
                print("Invalid Input: first index out of range.")
            break
        except ValueError:
            print("Invalid Input: Indices must be integers.")
    copy[int(location[1])][int(location[0])] = 2
    return copy, int(location[1]), int(location[0])


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


class Results(Enum):
    WIN = 1
    CAT = 0
    LOSS = -1  # Is this loss?


def SimulateGameRandom(realAgent, randomAgent, testing):
    board = np.zeros((3, 3), dtype=np.int8)
    if np.random.randint(0, 2) == 0:
        board, row_action, column_action = randomAgent.act(board)
        if CheckVictory(board, row_action, column_action):
            return Results.LOSS
        if CheckCat(board):
            return Results.CAT
    while True:
        if testing:
            board, row_action, column_action = realAgent.fill_board(
                realAgent.actBest(board), board, 1)
        else:
            board, row_action, column_action = realAgent.fill_board(
                realAgent.act(board), board, 1)
        if CheckVictory(board, row_action, column_action):
            return Results.WIN
        if CheckCat(board):
            return Results.CAT
        board, row_action, column_action = randomAgent.act(board)
        if CheckVictory(board, row_action, column_action):
            return Results.LOSS
        if CheckCat(board):
            return Results.CAT


def SimulateGameHuman(realAgent, personFunction):
    board = np.zeros((3, 3), dtype=np.int8)
    if np.random.randint(0, 2) == 0:
        board, row_action, column_action = personFunction(board)
        if CheckVictory(board, row_action, column_action):
            return Results.LOSS
        if CheckCat(board):
            return Results.CAT
    while True:
        board, row_action, column_action = realAgent.fill_board(
            realAgent.actBest(board), board, 1)
        if CheckVictory(board, row_action, column_action):
            return Results.WIN
        if CheckCat(board):
            return Results.CAT
        board, row_action, column_action = personFunction(board)
        if CheckVictory(board, row_action, column_action):
            return Results.LOSS
        if CheckCat(board):
            return Results.CAT


if __name__ == "__main__":
    print(DataFrame(board.transpose()))
    state_size = np.ravel(board).shape
    agentOne = Doe(state_size)
    agentTwo = Tac(2)
    EPISODES = 255168
    BATCH_SIZE = 32
    PRINT_RATE = 1000
    TEST_SIZE = 100
    FIGHT_RATE = 100000
    for e in range(EPISODES):
        result = SimulateGameRandom(agentOne, agentTwo, False)
        agentOne.remember_game(result.value)
        if (e+1) % BATCH_SIZE == 0:
            agentOne.train()
        if (e+1) % PRINT_RATE == 0:
            recent_wins = 0
            recent_losses = 0
            recent_cats = 0
            for i in range(TEST_SIZE):
                result = SimulateGameRandom(agentOne, agentTwo, True)
                if result == Results.WIN:
                    recent_wins += 1
                if result == Results.CAT:
                    recent_cats += 1
                if result == Results.LOSS:
                    recent_losses += 1
            print("Win %: {:.2}, Loss %: {:.2}, Cat %: {:.2}, E: {:.2}".format(recent_wins/TEST_SIZE,
                                                                               recent_losses/TEST_SIZE, recent_cats/TEST_SIZE, agentOne.exploration))
        if (e+1) % FIGHT_RATE == 0:
            SimulateGameHuman(agentOne, personAct)
