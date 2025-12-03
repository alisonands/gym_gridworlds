import numpy as np
import gym_gridworlds
import gymnasium as gym
import time
import matplotlib.pyplot as plt


env = gym.make("Gym-Gridworlds/FourRooms-Original-13x13-v0", 
               no_stay=True, 
               random_goals=False, 
               start_pos = None,
               distance_reward=True)#, render_mode = "human")

q = np.zeros((env.observation_space.n, env.action_space.n))

alpha = 0.01 # learning rate
gamma = 0.99 # discount factor 
eps = 0.01 # exploration
total_reward = 0
rewards_list = []

for episode in range(10000):
    obs, info = env.reset()

    done = False
    while not done:
        if np.random.rand() < eps:
            action = env.action_space.sample()
        else:
            action = np.argmax(q[obs])

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        q[obs, action] += alpha * (reward + gamma * np.max(q[next_obs]) - q[obs, action])
        obs = next_obs
        total_reward += reward

    rewards_list.append(total_reward)
    total_reward = 0
    print(f"Episode {episode}: Total Reward = {total_reward}")


env = gym.make("Gym-Gridworlds/FourRooms-Original-13x13-v0", 
               no_stay=True, 
               random_goals= False, 
               start_pos = None,
               distance_reward=True, 
               render_mode = "human")


for episode in range(10):
    obs, info = env.reset()
    done = False
    episode_reward = 0
    while not done:
        action = np.argmax(q[obs])
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        # time.sleep(0.1)
    print(f"Episode {episode + 1}: Total Reward = {episode_reward}")

env.close() 
plt.plot(rewards_list, '.')
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.show()

