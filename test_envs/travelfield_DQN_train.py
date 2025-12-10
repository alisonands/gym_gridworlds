#imports
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
import numpy as np
import gymnasium as gym
import gym_gridworlds
from gym_gridworlds.observation_wrappers import AddGoalWrapper, MatrixWithGoalWrapper
import os

# vars
env_name = "TravelField-10x10-v0"
save_model_name = "travelfield"
n_model = "DQN"
no_stay = True
distance_reward = True
start_pos = None
random_goals = False

# make env
env = gym.make(f"Gym-Gridworlds/{env_name}",
               no_stay = no_stay,
               distance_reward = distance_reward,
               start_pos = start_pos,
               random_goals = random_goals)

# make dirs
LOG_DIR = "log_dir/"
MODEL_DIR = "trained_models/"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# wrap envs
env = MatrixWithGoalWrapper(env)
env = Monitor(env, f"{LOG_DIR}") #logs stuff to log dir

# train
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=0.0009,
    gamma=0.916,
    exploration_fraction=0.07,
    verbose=1,
)

model.learn(total_timesteps=800000, progress_bar = True)

# save model
model.save(f"trained_models/{save_model_name}_{n_model}")