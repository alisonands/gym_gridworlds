import gymnasium
import gym_gridworlds
import pygame
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO


env = gymnasium.make("Gym-Gridworlds/Empty-RandomGoal-3x3-v0", render_mode="human")
obs, info = env.reset()



episode_over = False
total_reward = []
timestep_list = []
timestep = 0
current_reward = 0
step = 0

model = PPO.load("PPO_model")

while not episode_over:
    env.render()
    action, _ = model.predict(obs, deterministic=True)


    # for event in pygame.event.get():
    #     if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
    #         plt.plot(timestep_list, total_reward)
    #         plt.show()
    #     if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
    #         episode_over = True

    # action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    obs = observation
    # current_reward += reward
    # timestep += 1
    # print(timestep)
    # action, _ = model.predict(obs)
    # timestep_list.append(timestep)
    # total_reward.append(current_reward)
    episode_over = terminated or truncated
# env.close()