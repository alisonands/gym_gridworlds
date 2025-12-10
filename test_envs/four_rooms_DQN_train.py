#imports
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
import numpy as np
import gymnasium as gym
import gym_gridworlds
from gym_gridworlds.observation_wrappers import MatrixWithGoalWrapper
import os

# vars
env_name = "FourRooms-Original-13x13-v0"
save_model_name = "four_rooms"
n_model = "DQN"
no_stay = True
distance_reward = True
start_pos = None
random_goals = False

# wrapper class
class OneHotWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        n = env.observation_space.n
        self.observation_space = gym.spaces.Box(0, 1, (n,), dtype=np.float32)

    def observation(self, obs):
        v = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        v[obs] = 1.0
        return v

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
    learning_rate=0.0006,
    n_steps = 256, 
    batch_size=64,
    gamma=0.916,
    exploration_fraction=0.08,
    verbose=1,
)

model.learn(total_timesteps=50000, progress_bar = True)

# save model
model.save(f"trained_models/{save_model_name}_{n_model}")