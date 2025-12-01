import gymnasium as gym
import gym_gridworlds
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import pandas as pd
import os

# Create a directory to save logs
log_dir = "logs/"
os.makedirs(log_dir, exist_ok=True)

# create gridworld environment using gymnasium
env = gym.make("Gym-Gridworlds/CliffWalk-4x12-v0", no_stay=True, distance_reward = True)#, render_mode="human")

# Wrap the environment with a Monitor to log results
env = Monitor(env, log_dir)

# setup model and parameters
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=1e-4,
    # buffer_size=50000,
    # learning_starts=1000,
    batch_size=32,
    # tau=1.0,
    gamma=0.99,  #discount factor
    # train_freq=(1, "step"),
    # gradient_steps=1,
    # target_update_interval=250,
    # exploration_fraction=0.1,
    # exploration_final_eps=0.05,
    verbose=1
)

# train model, progress bar w tdqm
print("--- Starting Training ---")
model.learn(total_timesteps=50000, progress_bar=False)
print("--- Training Finished ---")

# save the model
model.save("ppo_gridworld_cliffwalk")
eval_env = gym.make("Gym-Gridworlds/CliffWalk-4x12-v0", no_stay=True) #, render_mode="human")

trained_model = PPO.load("ppo_gridworld_cliffwalk", env=eval_env)

mean_reward, std_reward = evaluate_policy(trained_model, eval_env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# --- Plotting the results ---
# monitor_file = os.path.join(log_dir, "monitor.csv")
# if os.path.exists(monitor_file):
#     # Load the log file
#     df = pd.read_csv(monitor_file, skiprows=1)
#     # Calculate cumulative timesteps
#     df['timesteps'] = df['l'].cumsum()
#     # Calculate a rolling average of the rewards
#     window_size = 50  # Average over the last 50 episodes
#     df['rolling_mean_reward'] = df['r'].rolling(window=window_size).mean()

#     # Plot original and smoothed rewards
#     fig, ax = plt.subplots()
#     df.plot(x='timesteps', y='r', kind='line', ax=ax, alpha=0.3, label='Episode Reward')
#     df.plot(x='timesteps', y='rolling_mean_reward', kind='line', ax=ax, color='red', label=f'Rolling Mean (window={window_size})')
    
#     ax.set_title('Reward over Timesteps')
#     ax.set_ylabel("Reward")
#     plt.show()

