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


def personAct(board):
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


if __name__ == "__main__":
    print(DataFrame(board.transpose()))
    state_size = np.ravel(board).shape
    agentOne = Doe(state_size)
    agentTwo = Doe(state_size)
    EPISODES = 1000
    BATCH_SIZE = 32
    for e in range(EPISODES):
        while True:
            board, row_action, column_action = agentOne.fill_board(
                agentOne.act(board), board, 1)
            if CheckVictory(board, row_action, column_action):
                agentOne.remember_game(1)
                agentTwo.remember_game(0)
                board = np.zeros((3, 3), dtype=np.int8)
                print("Agent One Wins")
                break
            if CheckCat(board):
                agentOne.remember_game(0.25)
                agentTwo.remember_game(0.25)
                board = np.zeros((3, 3), dtype=np.int8)
                print("Cat game")
                break
            board, row_action, column_action = agentTwo.fill_board(
                agentTwo.act(board), board, 2)
            if CheckVictory(board, row_action, column_action):
                agentOne.remember_game(0)
                agentTwo.remember_game(1)
                board = np.zeros((3, 3), dtype=np.int8)
                print("Agent two Wins")
                break
            if CheckCat(board):
                agentOne.remember_game(0.25)
                agentTwo.remember_game(0.25)
                board = np.zeros((3, 3), dtype=np.int8)
                print("Cat game")
                break
        if (e+1) % BATCH_SIZE == 0:
            agentOne.train()
            agentTwo.train()
            print("training")
    print("DOOOONEE")
    oneWins = 0
    oneCats = 0
    oneLosses = 0
    randAgentTwo = Tac(2)
    randAgentOne = Tac(1)

    def printResults(Wins, Cats, Losses):
        total = Wins + Cats + Losses
        print("Win %: {:.2}, Cat %: {:.2}, Loss %: {:.2}".format(
            Wins/total, Cats/total, Losses/total))
    while True:
        board, row_action, column_action = agentOne.fill_board(
            agentOne.act(board), board, 1)
        if CheckVictory(board, row_action, column_action):
            oneLosses += 1
            board = np.zeros((3, 3), dtype=np.int8)
        if CheckCat(board):
            oneCats += 1
            board = np.zeros((3, 3), dtype=np.int8)
        board, row_action, column_action = randAgentTwo.act(board)
        if CheckVictory(board, row_action, column_action):
            oneWins += 1
            board = np.zeros((3, 3), dtype=np.int8)
        if CheckCat(board):
            oneCats += 1
            board = np.zeros((3, 3), dtype=np.int8)
        if oneWins % 100 == 99:
            printResults(oneWins, oneCats, oneLosses)
    # while True:
    #     print(DataFrame(board.transpose()))
    #     print("")
    #     board, row_action, column_action = personAct(board)
    #     if CheckCat(board) or CheckVictory(board, row_action, column_action):
    #         board = np.zeros((3, 3), dtype=np.int8)
    #     print(DataFrame(board.transpose()))
    #     print("")
    #     board, row_action, column_action = agentOne.fill_board(
    #         agentOne.act(board), board, 1)
    #     if CheckCat(board) or CheckVictory(board, row_action, column_action):
    #         board = np.zeros((3, 3), dtype=np.int8)
