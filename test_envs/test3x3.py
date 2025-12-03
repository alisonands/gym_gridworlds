import gymnasium as gym
import gym_gridworlds
from stable_baselines3 import DQN, PPO
import time

env = gym.make("Gym-Gridworlds/Penalty-3x3-v0", start_pos = (0, 0), render_mode="human")
model = DQN.load("trained_models/DQN_3x3")

for episode in range(100):
    obs, info = env.reset()
    done = False
    episode_reward = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        time.sleep(0.1)
    print(f"Episode {episode + 1}: Total Reward = {episode_reward}")

env.close() 