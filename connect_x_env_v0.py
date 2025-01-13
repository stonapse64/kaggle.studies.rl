'''
Custom Gym environment
https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/
'''
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

import connect_x_game_v0 as cx
import numpy as np

# Register this module as a gym environment. Once registered, the id is usable in gym.make().
register(
    id='connect_x_env_v0',                   # call it whatever you want
    entry_point='connect_x_env_v0:ConnectXEnv', # module_name:class_namec
)


# Implement our own gym env, must inherit from gym.Env
# https://gymnasium.farama.org/api/env/
class ConnectXEnv(gym.Env):
    # metadata is a required attribute
    # render_modes in our environment is either None or 'human'.
    # render_fps is not used in our env, but we are require to declare a non-zero value.
    metadata = {"render_modes": ["human"], 'render_fps': 4}

    def __init__(self, rows: int = 6, columns: int = 7, inarow: int = 4, render_mode=None):
        # Ingest the game configuration       
        self.rows = rows
        self. columns = columns
        self.inarow = inarow

        self.size = self.rows * self.columns

        # Initialize the game
        self.game = cx.ConnectX(self.rows, self.columns, self.inarow)

        # Our observation is the 2d-board of size self.rows,self.columns
        self.observation_space = gym.spaces.Box(low=0, high=2, shape=(1,self.rows,self.columns), dtype=int)
        # We have 1 actions which is the column where the next piece goes
        self.action_space = gym.spaces.Discrete(self.columns)

        # StableBaselines throws error if these are not defined
        self.spec = None
        self.metadata = None
        
    # Gym required function (and parameters) to reset the environment
    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # gym requires this call to control randomness and reproduce scenarios.

        # Reset the game. Optionally, pass in seed control randomness and reproduce scenarios.
        self.obs = self.game.reset(seed=seed)
        
        # Additional info to return. For debugging or whatever.
        self.info = {}

        # Render environment
        if(self.render_mode=='human'):
            self.render()

        # Return observation and info
        return self.obs, self.info
    
    # Gym required function (and parameters) to perform an action
    def step(self, action):

        # Check if agent's move is valid
        is_valid = (self.game.obs['board'][int(action)] == 0)
        if is_valid: # Play the move
            self.obs, self.steps, self.reward, self.done, self.info = self.game.perform_action(action)
        else: # End the game and penalize agent for the invalid action
            self.done = True
        # Perform action
        # obs, step, reward, done, info = self.game.perform_action(action)

        self.terminated = self.done

        # Determine reward and termination
        if self.terminated:
            # This reward function penalizes early losses and late win as 
            # the reward from is cx.game.perform_action(action) is -1 for loss
            # and +1 for wins.
            reward_factor = (self.size - self.steps) / self.size
            self.reward = self.reward * reward_factor if is_valid else -1
            print(self.reward, reward_factor, self.steps)

        # Render environment
        if(self.render_mode=='human'):
            self.render()

        # Return observation, reward, terminated, truncated (not used), info
        return self.obs, self.reward, self.terminated, False, self.info 
    
    # Gym required function to render environment
    def render(self):
        self.game.render()
    
    def close(self):
        pass

# For unit testing
if __name__=="__main__":
    env = gym.make('connect_x_env_v0', render_mode='human')

    # # Use this to check our custom environment
    # print("Check environment begin")
    # check_env(env.unwrapped)
    # print("Check environment end")

    # Reset environment
    obs = env.reset()[0]

    # Take some random actions
    while(True):
        rand_action = env.action_space.sample()
        obs, reward, terminated, _, _ = env.step(rand_action)

        if(terminated):
            obs = env.reset()[0]