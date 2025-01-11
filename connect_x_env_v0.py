'''
Custom Gym environment
https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/
'''
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

import v0_warehouse_robot as wr
import numpy as np

# Register this module as a gym environment. Once registered, the id is usable in gym.make().
register(
    id='connect_x_env_v0',                   # call it whatever you want
    entry_point='connect_x_env_v0:ConnectXEnv', # module_name:class_name
)

