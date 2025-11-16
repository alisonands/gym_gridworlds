import gymnasium
import gym_gridworlds
import pygame
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO


env = gymnasium.make("Gym-Gridworlds/Corridor-3x4-v0")#, render_mode="human")
obs, info = env.reset()

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)


episode_over = False
total_reward = []
timestep_list = []
timestep = 0
current_reward = 0
step = 0

for _ in range(100):
    # for event in pygame.event.get():
    #     if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
    #         plt.plot(timestep_list, total_reward)
    #         plt.show()
    #     if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
    #         episode_over = True

    action = env.action_space.sample()
    observation, reward, _, _, _ = env.step(action)
    # current_reward += reward
    # timestep += 1
    action, _ = model.predict(obs)
    # timestep_list.append(timestep)
    # total_reward.append(current_reward)
    # episode_over = terminated or truncated
    