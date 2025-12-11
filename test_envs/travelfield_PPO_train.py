# ---------------------------------
# IMPORTS
# ---------------------------------
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import numpy as np
import os

# custom environment
import gym_gridworlds
from gym_gridworlds.observation_wrappers import AddGoalWrapper, MatrixWithGoalWrapper

# ---------------------------------
# VARIABLES (change as required)
# ---------------------------------
env_name = "TravelField-10x10-v0"
save_model_name = "travelfield"
n_model = "PPO"
no_stay = True
start_pos = None
random_goals = False
distance_reward = True

# ---------------------------------
# CREATE ENV (gymnasium)
# ---------------------------------
env = gym.make(f"Gym-Gridworlds/{env_name}",
               no_stay = no_stay,
               start_pos = start_pos,
               random_goals = random_goals,
               distance_reward = distance_reward)

# make dirs
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
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=0.01, 
    gamma=0.99,
    ent_coef=0.1,
    verbose=1,
)

model.learn(total_timesteps=100000, progress_bar = True)

# save model
model.save(f"trained_models/{save_model_name}_{n_model}")