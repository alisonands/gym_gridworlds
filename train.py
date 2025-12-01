# %%
# imports 
from stable_baselines3 import DQN, PPO
import gymnasium as gym
import gym_gridworlds
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3.common.evaluation import evaluate_policy
import os
import torch.nn as nn
import numpy as np

# %%
# vars
env_name = "TravelField-10x10-v0"
save_model_name = "travelfield"
distance_reward = True
no_stay = True
start_pos = (0,0)
random_goals = False

# %%
# make environment
env = gym.make(f"Gym-Gridworlds/{env_name}", 
               no_stay = no_stay, 
               distance_reward = distance_reward, 
               start_pos = start_pos, 
               random_goals = random_goals)
# logging wrapper
os.makedirs("logs/", exist_ok=True)
os.makedirs("trained_models", exist_ok=True)

# %%
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=1e-4,
    n_steps=64,
    batch_size=64,
    # buffer_size=50000,
    # learning_starts=1000,
    # batch_size=32,
    # tau=1.0,
    gamma=0.99,
    # train_freq=(1, "step"),
    # gradient_steps=1,
    # target_update_interval=250,
    exploration_fraction=0.5,
    # exploration_final_eps=0.05,
    verbose=1,
)
# model = PPO(
#     "MlpPolicy",
#     env,
#     learning_rate=1e-4,
#     n_steps=64,
#     batch_size=64,
#     gamma=0.95,
#     gae_lambda=0.95,
#     clip_range=0.2,
#     clip_range_vf=0.4,
#     ent_coef=0.05,
#     vf_coef=0.7,
#     target_kl=0.02,
#     verbose=1,
#     normalize_advantage=True,
#     policy_kwargs = dict(
#     net_arch=[32, 32],
#     activation_fn=nn.Tanh,
# )
# )

# %%
# train model, progress bar w tdqm
print("--- Starting Training ---")
model.learn(total_timesteps=50000, progress_bar=False)
print("--- Training Finished ---")

# %%
model.save(f"test_envs/trained_models/{save_model_name}")

# %%
eval_env = gym.make(f"Gym-Gridworlds/{env_name}", no_stay=no_stay, distance_reward=distance_reward)
trained_model = DQN.load(f"test_envs/trained_models/{save_model_name}")
mean_reward, std_reward = evaluate_policy(trained_model, eval_env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")


# %% [markdown]
# ## Notes
# - Default timestep = 500 per run


