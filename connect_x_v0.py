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
    RANDOMSELECT = ''
    RANDOM = 'random'
    NEGAMAX = 'negamax'


class ConnectX:
    def __init__(self, rows: int = 6, columns: int = 7, inarow: int = 4, fps=1):
        self.rows = rows
        self. columns = columns
        self.inarow = inarow
        self.fps=fps
        self.env = make("connectx", {"rows": self.rows, "columns": self.columns,
                                      "inarow": 4}, debug=True)

        self.trainer = self.trainer_choice()
        self.obs = self.trainer.reset()

    def reset(self, seed=None):
        # self.trainer = self.trainer_choice(True, True)
        self.obs = self.trainer.reset()
    
    def perform_action(self, column):
        self.obs, self.reward, self.done, self.info = self.trainer.step(column)
        return self.obs, self.reward, self.done, self.info
        pass
    
    def render(self):
        self.env.render(mode="ipython")
        pass
    
    def trainer_choice(self, opponent_choice="", player_to_start=0):
        assert opponent_choice in ConnectXOpponents
        assert player_to_start in [0, 1, 2]

        # Selecting a the opponent agent.
        opponent = random.choice([*self.env.agents]) if opponent_choice == "" \
            else opponent_choice
        
        # Selecting a the playing order.
        player_to_start = random.randint(1, 2) if player_to_start == 0 \
            else player_to_start
        
        # Setting the opponent agent and the playing order.        
        trainer = self.env.train([None, opponent]) if player_to_start == 1 \
                else self.env.train([opponent, None])
        print(f"You are player {player_to_start} playing against {opponent}.")

        return trainer


# For unit testing
if __name__=="__main__":
    connectx = ConnectX(rows=6, columns=7)

    episodes = 5    

    for i in range(episodes):

        episode_over = False

        while not episode_over:
            rand_action = random.randint(0, connectx.columns - 1)
            # print(rand_action)
            obs, reward, done, info = connectx.perform_action(rand_action)
            if done:
                if reward == 1: print("You won!")
                elif reward == 0: print("Just a draw?")
                else: print("You lost...")
                episode_over = True
                connectx.reset(seed=None)

        connectx.render()