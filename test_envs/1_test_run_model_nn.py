# ---------------------------------
# IMPORTS
# ---------------------------------

import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
import numpy as np
import time

# custom environment
import gym_gridworlds
from gym_gridworlds.observation_wrappers import AddGoalWrapper

# ---------------------------------
# VARIABLES (change as required)
# ---------------------------------

env_name = "TravelField-10x10-v0"
save_model_name = "travelfield_DQN"
no_stay = True
start_pos = None
random_goals = False
distance_reward = True


# ---------------------------------
# CREATE ENV (gymnasium)
# ---------------------------------
env = gym.make(f"Gym-Gridworlds/{env_name}", 
               no_stay=no_stay, 
               distance_reward=distance_reward, 
               start_pos=start_pos, 
               random_goals=random_goals,
               render_mode="human")
# logging for reward curves
LOG_DIR = "log_dir/"
env = AddGoalWrapper(env) # add goal observation
env = Monitor(env, f"{LOG_DIR}") 

# ---------------------------------
# MODEL EVALUATION 
# ---------------------------------
model = DQN.load(f"trained_models/{save_model_name}")

for episode in range(2):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _states = model.predict(np.array(obs))
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        # time.sleep(0.1) # <-- for slow(er) rendering
env.close()