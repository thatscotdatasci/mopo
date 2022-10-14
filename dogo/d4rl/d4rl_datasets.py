import gym
import d4rl
import numpy as np


##########################################################
# Capture D4RL datasets and save them in my desired format
##########################################################


name = 'halfcheetah-random-v0'
env = gym.make(name)
data = d4rl.qlearning_dataset(env)
data_arr = np.hstack([
    data['observations'],
    data['actions'],
    data['next_observations'],
    data['rewards'][:,None],
    data['terminals'][:,None],
    np.zeros_like(data['rewards'])[:,None]
])
# np.save('../dogo_results/data/D4RL-HC-R.npy', data_arr)
