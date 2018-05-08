from pandas import *
from pprint import pprint
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, Input
from keras.losses import categorical_crossentropy
from keras.optimizers import rmsprop
import math
import time
import random

board = np.zeros((3, 3), dtype=np.int8)
magic_number = 9


def personAct():
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
    board[int(location[0])][int(location[1])] = 1


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


class Doe():
    def __init__(self, state_shape, hyperparams={'max_len': 2000, 'batch_size': 32}):
        self.input_shape = state_shape
        self.model = self._build_model()
        self.batch_size = hyperparams['batch_size']
        self.memory = deque(maxlen=hyperparams['max_len'])
        self.short_term = deque(maxlen=5)

    def _build_model(self):
        model = Sequential()
        model.add(Dense(9, activation='relu',
                        kernel_initializer='random_uniform', input_shape=self.input_shape))
        model.add(Dense(9, activation='relu',
                        kernel_initializer='random_uniform'))
        model.add(Dense(9, activation='relu',
                        kernel_initializer='random_uniform'))
        model.add(Dense(9, activation='relu',
                        kernel_initializer='random_uniform'))
        model.add(Dense(9, activation='relu',
                        kernel_initializer='random_uniform'))
        model.add(Dense(9, activation='relu',
                        kernel_initializer='random_uniform'))
        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def act(self, state):
        outcomes = self.model.predict(self.processState(state))
        # print(state)
        while True:
            selected = np.argmax(outcomes[0])
            row_action = (selected) % 3
            column_action = math.ceil((selected+1)/3) - 1
            if state[row_action][column_action] == 0:
                break
            else:
                temp = np.argmax(outcomes[0])
                outcomes[0][temp] -= 1
                # print(state)
        self.short_term.append((state, selected))
        return (selected+1)

    def fill_board(self, action, board, thisIndex):
        row_action = (action-1) % 3
        column_action = math.ceil(action/3) - 1
        board[row_action][column_action] = thisIndex
        return board, row_action, column_action

    def remember_game(self, victory):
        for state, selected in self.short_term:
            self.memory.append((state, selected, victory))
        self.short_term.clear()

    def train(self):
        # print(self.memory)
        training_data = random.sample(list(self.memory), self.batch_size)
        states, actions, rewards = zip(*training_data)
        states, actions, rewards = list(states), list(actions), list(rewards)
        print(len(states))
        target_rewards = []
        for state in states:
            target_rewards.append(self.model.predict(self.processState(state)))
        print(np.array(target_rewards).shape)
        for i in range(self.batch_size):
            print(target_rewards[i])
            target_rewards[i][0][actions[i]] = rewards[i]
            states[i] = self.processState(states[i])
        self.model.fit(states, target_rewards, epochs=1, verbose=0)

    def processState(self, array):
        return array.reshape(1, magic_number)


if __name__ == "__main__":
    print(DataFrame(board.transpose()))
    state_size = np.ravel(board).shape
    agentOne = Doe(state_size)
    agentTwo = Doe(state_size)
    EPISODES = 100
    BATCH_SIZE = 32
    for e in range(EPISODES):
        while True:
            # print("moving")
            # print(DataFrame(board.transpose()))
            # time.sleep(1)
            board, row_action, column_action = agentOne.fill_board(
                agentOne.act(board), board, 1)
            if CheckCat(board):
                agentOne.remember_game(0)
                agentTwo.remember_game(0)
                board = np.zeros((3, 3), dtype=np.int8)
                print("Cat game")
                break
            if CheckVictory(board, row_action, column_action):
                agentOne.remember_game(1)
                agentTwo.remember_game(0)
                board = np.zeros((3, 3), dtype=np.int8)
                print("Agent One Wins")
                break
            board, row_action, column_action = agentTwo.fill_board(
                agentTwo.act(board), board, 2)
            if CheckCat(board):
                agentOne.remember_game(0)
                agentTwo.remember_game(0)
                board = np.zeros((3, 3), dtype=np.int8)
                print("Cat game")
                break
            if CheckVictory(board, row_action, column_action):
                agentOne.remember_game(0)
                agentTwo.remember_game(1)
                board = np.zeros((3, 3), dtype=np.int8)
                print("Agent two Wins")
                break
        if (e+1) % BATCH_SIZE == 0:
            agentOne.train()
            agentTwo.train()
            print("training")
