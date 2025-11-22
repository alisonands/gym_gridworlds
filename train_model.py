import gymnasium
import gym_gridworlds
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO


env = gymnasium.make("Gym-Gridworlds/Full-4x5-v0")#, render_mode="human")
obs, info = env.reset()

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)
model.save('PPO_model')

