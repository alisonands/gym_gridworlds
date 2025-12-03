import gymnasium as gym
import gym_gridworlds
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
import numpy as np
import time


# vars
env_name = "TravelField-10x10-v0"
save_model_name = "travelfield_DQN"
distance_reward = True
no_stay = True
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

env = gym.make(f"Gym-Gridworlds/{env_name}", 
               no_stay=no_stay, 
               distance_reward=distance_reward, 
               start_pos=start_pos, 
               random_goals=random_goals,
               render_mode="human")
env = OneHotWrapper(env)

# load model
model = DQN.load(f"trained_models/{save_model_name}")

for episode in range(10):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        # time.sleep(0.1)
env.close()