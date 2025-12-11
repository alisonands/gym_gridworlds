import numpy as np
import gym_gridworlds
import gymnasium as gym
import time
import matplotlib.pyplot as plt

# ---------------------------------
# CREATE ENV (gymnasium)
# ---------------------------------
env = gym.make("Gym-Gridworlds/TravelField-10x10-v0", 
               no_stay=True, 
               random_goals=False, 
               start_pos = None,
               distance_reward=True)#, render_mode = "human")

# ---- define observation and action spaces -----
q = np.zeros((env.observation_space.n, env.action_space.n))

alpha = 0.08 # learning rate
gamma = 0.94 # discount factor 
eps = 0.06 # exploration
rewards_list = []

for episode in range(10000):
    obs, info = env.reset()

    done = False
    episode_reward = 0
    while not done:
        if np.random.rand() < eps:
            action = env.action_space.sample()
        else:
            action = np.argmax(q[obs])

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # ------ Q-learning update function -------
        q[obs, action] += alpha * (reward + gamma * np.max(q[next_obs]) - q[obs, action])
        obs = next_obs
        episode_reward += reward

    rewards_list.append(episode_reward)
    print(f"Episode {episode}: Total Reward = {episode_reward}")


env = gym.make("Gym-Gridworlds/TravelField-10x10-v0", 
               no_stay=True, 
               random_goals= False, 
               start_pos = None,
               distance_reward=True, 
               render_mode = "human")


eval_rewards = [] 
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
    eval_rewards.append(episode_reward)
    print(f"Episode {episode + 1}: Total Reward = {episode_reward}")

mean_reward = np.mean(eval_rewards)
std_reward = np.std(eval_rewards)
print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

env.close() 

# Calculate moving average for a smoother learning curve
window_size = 100
moving_avg = np.convolve(rewards_list, np.ones(window_size)/window_size, mode='valid')

plt.plot(np.arange(window_size - 1, len(rewards_list)), moving_avg, label=f'{window_size}-Episode Moving Average')
plt.xlabel("Number of Timesteps")
plt.ylabel("Rewards")
plt.title("Learning Curve Travel Field Q_learning")
plt.grid(True)
plt.savefig("plots/travelfield_Q_learning.png")
plt.show()
