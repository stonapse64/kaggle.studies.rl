'''
This module models the problem to be solved. In this  example, the problem is to
optimze a Connect X agent that plays Connect X against a randomly selected 
type of opponent. The player to start the is also randomly selected.
The playing field is divided into columns and rows. The default is size of 7 by 
6. The player alternately drop a piece into the grid and it falls down as far as
possible. The game is won when n pieces of a player are either horizontally, 
vertically or diagonally in a sequence that is not interrupted by an opponent's
piece. The game ends with a draw when all places of the playing field are filled
with pieces and neither the agent nor the opponent accomplished a winning 
sequence.
'''

import random
from enum import Enum
import pygame
import sys
from os import path
from kaggle_environments import make


class ConnectXOpponents(Enum):
    pass


class ConnectX:
    def __init__(self, rows=6, columns=7, inarow=4, fps=1):
        self.rows = rows
        self. columns = columns
        self.size = self.rows * self.columns
        self.fps=fps
        self.env = make("connectx", {"rows": self.rows, "columns": self.columns, "inarow": 4}, debug=False)

        self.trainer = self.trainer_choice(True, True)
        self.obs = self.trainer.reset()
        # self.render()

    def reset(self, seed=None):
        # self.trainer = self.trainer_choice(True, True)
        self.obs = self.trainer.reset()
    def perform_action(self, column) -> bool:
        self.obs, self.reward, self.done, self.info = self.trainer.step(column)
        return self.obs, self.reward, self.done, self.info
        pass
    def render(self):
        # self.env.render(mode="ipython")
        pass
    def trainer_choice(self, random_opponent=True, random_player_starts=True):
        if random_opponent:
            opponent = random.choice([*self.env.agents])
            print("You are playing against the default player:", opponent)
        if random_player_starts:
            if random.randint(0, 1):
            # Training agent in first position (player 1) against a random default agent.
                print("You will start the game.")
                trainer = self.env.train([None, opponent])                
            else:
                # Training agent in second position (player 2) against a random default agent.
                trainer = self.env.train([opponent, None])
                print("Your opponent", opponent, "will start the game.")
        return trainer
    

# For unit testing
if __name__=="__main__":
    connectx = ConnectX(rows=10, columns=10)
    connectx.render()

    running = True

    while(running):
        rand_action = random.randint(0, connectx.columns - 1)
        print(rand_action)
        obs, reward, done, info = connectx.perform_action(rand_action)
        connectx.render()
        if done:
            if reward == 1: print("you won")
            elif reward == 0: print("a draw")
            else: print("you lost")
            running = False

