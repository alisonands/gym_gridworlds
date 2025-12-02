import gymnasium as gym
import gym_gridworlds
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor
import numpy as np
import time


# vars
env_name = "FourRooms-Original-13x13-v0"
save_model_name = "four_rooms"
distance_reward = True
no_stay = True
start_pos = None
random_goals = False

env = gym.make(f"Gym-Gridworlds/{env_name}", 
               no_stay=no_stay, 
               distance_reward=distance_reward, 
               start_pos=start_pos, 
               random_goals=random_goals,
               render_mode="human")
env = Monitor(env, "logs/")
model = DQN.load(f"trained_models/{save_model_name}")

def one_hot(obs, n):
    v = np.zeros(n, dtype=np.float32)
    v[obs] = 1.0
    return v

for episode in range(10):
    obs, info = env.reset()
    obs = one_hot(obs, env.observation_space.n)
    done = False
    episode_reward = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        new_obs, reward, terminated, truncated, info = env.step(action)
        obs = one_hot(new_obs, env.observation_space.n)
        done = terminated or truncated
        episode_reward += reward
        time.sleep(0.1)
    print(f"Episode {episode + 1}: Total Reward = {episode_reward}")

env.close() 