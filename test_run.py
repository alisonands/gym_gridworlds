import gymnasium as gym
import gym_gridworlds
import pygame
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy

model = PPO.load("ppo_gridworld_cliffwalk")
eval_env = gym.make("Gym-Gridworlds/CliffWalk-4x12-v0", no_stay=True, render_mode="human")

# Load the trained agent
trained_model = PPO.load("ppo_gridworld_cliffwalk", env=eval_env)

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(trained_model, eval_env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")


for episode in range(5):
    obs, info = eval_env.reset()
    done = False
    episode_reward = 0
    while not done:
        action, _states = trained_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated
        episode_reward += reward
    print(f"Episode {episode + 1}: Total Reward = {episode_reward}")





# env = gymnasium.make("Gym-Gridworlds/Full-4x5-v0", render_mode="human")
# obs, info = env.reset()



# episode_over = False
# total_reward = []
# timestep_list = []
# timestep = 0
# current_reward = 0
# step = 0

# model = PPO.load("PPO_model")

# for i in range(1000):
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             pygame.quit()
#             quit()
#     action, _ = model.predict(obs, deterministic=True) # if deterministic --> uses stochastic policy 
#     obs, reward, terminated, truncated, info = env.step(action)
#     timestep += 1
#     current_reward += reward
#     print(f'timestep: {timestep}, reward: {current_reward}')
#     episode_over = terminated or truncated
#     if episode_over:
#         # print(f"Episode finished. Total reward: {current_reward}")
#         # total_reward.append(current_reward)
#         # Reset for next episode
#         timestep = 0
#         current_reward = 0
#         obs, info = env.reset()
        
# env.close()