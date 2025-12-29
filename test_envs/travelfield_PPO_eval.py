# plotting reward curves

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
import numpy as np
import os

# custom environment
import gym_gridworlds
from gym_gridworlds.observation_wrappers import AddGoalWrapper

# ---------------------------------
# VARIABLES (change as required)
# ---------------------------------
env_name = "TravelField-10x10-v0"
save_model_name = "travelfield"
n_model = "PPO"
eval_model_name = f"{save_model_name}_eval_{n_model}"
no_stay = True
start_pos = None
random_goals = False
distance_reward = True

LOG_DIR = "log_dir/"

# ---------------------------------
# PLOTTING (change as required)
# ---------------------------------
# Plotting the learning curve from the Monitor logs
# Helper function from Stable Baselines 3 to read monitor files
def plot_results(log_folder, title=f'Learning Curve {save_model_name} {n_model}'):
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    # Smooth the curve using a moving average
    y = np.convolve(y, np.ones(100)/100, mode='valid')
    x = x[len(x) - len(y):]
    
    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title)
    plt.grid(True)
    plt.savefig(f"plots/{save_model_name}_{n_model}.png")
    plt.show()

plot_results(f"{LOG_DIR}")


eval_env = gym.make(f"Gym-Gridworlds/{env_name}", 
                    no_stay=no_stay, 
                    distance_reward=distance_reward,
                    start_pos = start_pos,
                    random_goals = random_goals,
                    )
# wrap envs
eval_env = AddGoalWrapper(eval_env)
eval_env = Monitor(eval_env) #, f"{LOG_DIR}/{eval_model_name}")

trained_model = PPO.load(f"trained_models/{save_model_name}_{n_model}")
mean_reward, std_reward = evaluate_policy(trained_model, eval_env, n_eval_episodes=10)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")