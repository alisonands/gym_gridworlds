# ---------------------------------
# IMPORTS
# ---------------------------------
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
import numpy as np
import os

# custom environment
import gym_gridworlds
from gym_gridworlds.observation_wrappers import AddGoalWrapper

# ---------------------------------
# VARIABLES (change as required)
# ---------------------------------
env_name = "FourRooms-Original-13x13-v0"
save_model_name = "four_rooms"
n_model = "DQN"
no_stay = True
start_pos = None
random_goals = False
distance_reward = True


# ---------------------------------
# CREATE ENV (gymnasium)
# ---------------------------------
env = gym.make(f"Gym-Gridworlds/{env_name}",
               no_stay = no_stay,
               distance_reward = distance_reward,
               start_pos = start_pos,
               random_goals = random_goals)


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
    learning_rate=0.023,
    gamma=0.955,
    exploration_fraction=0.07,
    verbose=1,
)

model.learn(total_timesteps=150000, progress_bar = True)

# save model
model.save(f"trained_models/{save_model_name}_{n_model}")