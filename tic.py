from pandas import *
from pprint import pprint
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.losses import categorical_crossentropy
from keras.optimizers import rmsprop

board = [[0 for y in range(3)] for x in range(3)]


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
    def __init__(self, state_shape, hyperparams={'max_len': 2000}):
        self.input_shape = state_shape
        self.model = self._build_model()
        self.memory = deque(maxlen=hyperparams['max_len'])
        self.short_term = deque(maxlen=5)

    def _build_model(self):
        model = Sequential()
        model.add(Input(shape=self.input_shape))
        model.add(Flatten())
        model.add(Dense(9, activation='relu'))
        model.add(Dense(9, activation='relu'))
        model.add(Dense(9, activation='relu'))
        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def act(self, state):
        outcomes = self.model.predict(state)
        selected = np.argmax(outcomes[0])
        self.short_term.append((state, selected))
        return selected

    def fill_board(self, action, board, thisIndex):
        row_action = (action-1) % 3
        column_action = (action/3) - 1
        board[column_action][row_action] = thisIndex
        return board, row_action, column_action

    def remember_game(self, victory):
        for state, selected in self.short_term:
            self.memory.append((state, action, victory))
        self.short_term.clear()

    def train():


if __name__ == "__main__":
    print(DataFrame(board))
    state_size = np.array(board).shape
    agentOne = Doe(state_size)
    agentTwo = Doe(state_size)
    EPISODES = 100
    BATCH_SIZE = 32
    for e in EPISODES:
        while True:
            board, row_action, column_action = agentOne.fill_board(
                agentOne.act(board), board, 1)
            if CheckVictory(row_action, column_action):
                agentOne.remember_game(1)
                agentOne.remember_game(0)
            board, row_action, column_action = agentTwo.fill_board(
                agentTwo.act(board), board, 2)
            if CheckVictory(row_action, column_action):
                agentOne.remember_game(0)
                agentOne.remember_game(1)
        if e % BATCH_SIZE == 0:
