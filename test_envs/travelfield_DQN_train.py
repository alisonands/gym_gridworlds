# training model on environment

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
import numpy as np
import os

# custom environments
import gym_gridworlds
from gym_gridworlds.observation_wrappers import AddGoalWrapper

# ---------------------------------
# VARIABLES (change as required)
# ---------------------------------
env_name = "TravelField-10x10-v0"
save_model_name = "travelfield"
n_model = "DQN"
no_stay = True
start_pos = None
random_goals = False
distance_reward = True

# make env
env = gym.make(f"Gym-Gridworlds/{env_name}",
               no_stay = no_stay,
               start_pos = start_pos,
               random_goals = random_goals,
               distance_reward = distance_reward)

# create dirs for reward curves and training models
LOG_DIR = "log_dir/"
MODEL_DIR = "trained_models/"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# wrap envs
env = AddGoalWrapper(env)
env = Monitor(env, f"{LOG_DIR}") #logs stuff to log dir

# -------------------------------------------
# TRAIN (parameters determined using optuna)
# -------------------------------------------
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=0.0001,
    gamma=0.93,
    exploration_fraction=0.1,
    verbose=1,
)

model.learn(total_timesteps=800000, progress_bar = True)

# save model
model.save(f"trained_models/{save_model_name}_{n_model}")