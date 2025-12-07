import gym_gridworlds
import gymnasium as gym

env = gym.make("Gym-Gridworlds/TravelField-10x10-v0", render_mode = "human")
obs, _ = env.reset()

episode_over = False

while not episode_over:
    action = env.action_space.sample()

    observation, info, terminated, truncated, _ = env.step(action)

    episode_over = terminated or truncated

env.close()

