import gymnasium
import gym_gridworlds
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO


env = gymnasium.make("Gym-Gridworlds/Empty-RandomGoal-3x3-v0")#, render_mode="human")
obs, info = env.reset()

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=25000)
model.save('PPO_model')