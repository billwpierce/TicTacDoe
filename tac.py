import numpy as np
import math
import time
import random


class Tac():
    def __init__(self, number):
        self.number = number

    def act(self, board):
        row_action = -1
        column_action = -1
        while True:
            row_action = random.randint(0, 2)
            column_action = random.randint(0, 2)
            if board[row_action][column_action] == 0:
                break
        board[row_action][column_action] = self.number
        return board, row_action, column_action
