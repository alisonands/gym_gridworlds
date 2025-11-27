import gymnasium as gym
import gym_gridworlds
from stable_baselines3 import DQN, PPO

env = gym.make("Gym-Gridworlds/Empty-RandomStart-3x3-v0", render_mode="human")
model = PPO.load("test_envs/trained_models/3x3")

for episode in range(5):
    obs, info = env.reset()
    done = False
    episode_reward = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_reward += reward
    print(f"Episode {episode + 1}: Total Reward = {episode_reward}")

env.close() 