import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, Input
from keras.losses import categorical_crossentropy
from keras.optimizers import rmsprop
from collections import deque
import math
import time
import random

magic_number = 9


class Doe():
    def __init__(self, state_shape, hyperparams={'max_len': 2000, 'batch_size': 32, 'exploration_init': 1.0, 'exploration_fin': 0.005, 'exploration_decay': 0.9999}):
        self.input_shape = state_shape
        self.model = self._build_model()
        self.batch_size = hyperparams['batch_size']
        self.memory = deque(maxlen=hyperparams['max_len'])
        self.short_term = deque(maxlen=5)
        self.exploration_init = hyperparams['exploration_init']
        self.exploration_decay = hyperparams['exploration_decay']
        self.exploration_fin = hyperparams['exploration_fin']
        self.exploration = self.exploration_init

    def _build_model(self):
        print(self.input_shape)
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
        selected = 0
        if self.exploration > np.random.rand():  # If we are exploring
            while True:
                selected = random.randint(0, 8)
                row_action = (selected) % 3
                column_action = math.ceil((selected+1)/3) - 1
                if state[row_action][column_action] == 0:
                    break
        else:
            outcomes = self.model.predict(self.processState(state))
            while True:
                selected = np.argmax(outcomes[0])
                row_action = (selected) % 3
                column_action = math.ceil((selected+1)/3) - 1
                if state[row_action][column_action] == 0:
                    break
                else:
                    temp = np.argmax(outcomes[0])
                    outcomes[0][temp] -= 1
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
        training_data = random.sample(list(self.memory), self.batch_size)
        states, actions, rewards = zip(*training_data)
        states, actions, rewards = list(states), list(actions), list(rewards)
        target_rewards = []
        for state in states:
            target_rewards.append(self.model.predict(
                self.processState(state))[0])
        for i in range(self.batch_size):
            target_rewards[i][actions[i]] = rewards[i]
            states[i] = self.processState(states[i])
        self.model.fit(np.array(states).reshape(32, 9), np.array(
            target_rewards), epochs=1, verbose=0)
        if self.exploration > self.exploration_fin:
            self.exploration *= self.exploration_decay

    def processState(self, array):
        return array.reshape(1, magic_number)
