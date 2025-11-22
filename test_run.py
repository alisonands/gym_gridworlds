import gymnasium
import gym_gridworlds
import pygame
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO


env = gymnasium.make("Gym-Gridworlds/Full-4x5-v0", render_mode="human")
obs, info = env.reset()



episode_over = False
total_reward = []
timestep_list = []
timestep = 0
current_reward = 0
step = 0

model = PPO.load("PPO_model")

for i in range(1000):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    timestep += 1
    current_reward += reward
    print(f'timestep: {timestep}, reward: {current_reward}')
    episode_over = terminated or truncated
    if episode_over:
        # print(f"Episode finished. Total reward: {current_reward}")
        # total_reward.append(current_reward)
        # Reset for next episode
        timestep = 0
        current_reward = 0
        obs, info = env.reset()
        
env.close()