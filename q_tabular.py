import numpy as np
import gym_gridworlds
import gymnasium as gym

env = gym.make("Gym-Gridworlds/Empty-RandomStart-3x3-v0", no_stay=True, distance_reward=True, render_mode = "human")

q = np.zeros((env.observation_space.n, env.action_space.n))

alpha = 0.1 # learning rate
gamma = 0.99 # discount factor 
eps = 0.2 # exploration

for episode in range(5000):
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

env.close() 
