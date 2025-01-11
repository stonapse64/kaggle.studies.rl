'''
Custom Gym environment
https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/
'''
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

import connect_x_v0 as cx
import numpy as np

# Register this module as a gym environment. Once registered, the id is usable in gym.make().
register(
    id='connect_x_env_v0',                   # call it whatever you want
    entry_point='connect_x_env_v0:ConnectXEnv', # module_name:class_name
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

        # TODO
        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`-1}^2
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 1 actions which is the column where the next piece goes
        self.action_space = gym.spaces.Discrete(self. columns - 1)
        
        # TODO
        # Dictionary maps the abstract actions to the directions on the grid
        self._action_to_direction = {
            0: np.array([1, 0]),  # right
            1: np.array([0, 1]),  # up
            2: np.array([-1, 0]),  # left
            3: np.array([0, -1]),  # down
        }

    # Gym required function (and parameters) to reset the environment
    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # gym requires this call to control randomness and reproduce scenarios.

        # Reset the WarehouseRobot. Optionally, pass in seed control randomness and reproduce scenarios.
        obs = self.game.reset(seed=seed)
        
        # Additional info to return. For debugging or whatever.
        info = {}

        # Render environment
        if(self.render_mode=='human'):
            self.render()

        # Return observation and info
        return obs, info
    
    # Gym required function (and parameters) to perform an action
    def step(self, action):
        # Perform action
        obs, reward, done, info = cx.game.perform_action(action)

        terminated = done

        # Determine reward and termination
        if terminated:
            # This reward function penalizes early losses and late win as 
            # the reward from is cx.game.perform_action(action) is -1 for loss
            # and +1 for wins.
            reward_factor = (self.size - obs.get('step', 0)) / self.size
            reward = reward * reward_factor

        # Render environment
        if(self.render_mode=='human'):
            self.render()

        # Return observation, reward, terminated, truncated (not used), info
        return obs, reward, terminated, False, info
    
    
    # Gym required function to render environment
    def render(self):
        self.game.render()
