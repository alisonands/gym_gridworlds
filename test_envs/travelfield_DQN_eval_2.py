# ---------------------------------
# IMPORTS
# ---------------------------------
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
import matplotlib.pyplot as plt
import numpy as np

# custom environment
import gym_gridworlds
from gym_gridworlds.observation_wrappers import AddGoalWrapper

# ---------------------------------
# VARIABLES (change as required)
# ---------------------------------
env_name = "TravelField-10x10-v0"
save_model_name = "travelfield"
n_model = "DQN"
eval_model_name = f"{save_model_name}_eval_{n_model}"
no_stay = True
start_pos = None
random_goals = False
distance_reward = True

LOG_DIR = "log_dir/"

# Visualize the learned policy and value function
trained_model = DQN.load(f"trained_models/{save_model_name}_{n_model}")
grid_size = 10
policy_grid = np.zeros((grid_size, grid_size), dtype=int)
value_grid = np.zeros((grid_size, grid_size))

# Visualize the learned policy and value function
grid_size = 10
trained_model = DQN.load(f"trained_models/{save_model_name}_{n_model}")
policy_grid = np.zeros((grid_size, grid_size), dtype=int)
value_grid = np.zeros((grid_size, grid_size))

goal_pos = 2 * grid_size + 9 # Goal is at (11, 11) for FourRooms-Original-13x13-v0

# Action mapping for arrows
action_arrows = {0: '←', 1: '↓', 2: '→', 3: '↑'}
for i in range(grid_size):
    for j in range(grid_size):
        obs = i * grid_size + j

        # format given by ActionGoalWrapper; two discrete positions
        wrapped_obs = np.array([obs, goal_pos])
        action, _ = trained_model.predict(wrapped_obs, deterministic=True)
        policy_grid[i, j] = action
        
        # For DQN, the state-value is the max of the Q-values for that state
        obs_tensor, _ = trained_model.policy.obs_to_tensor(np.array([wrapped_obs]))
        q_values = trained_model.q_net(obs_tensor)
        # The value of a state is the maximum Q-value for that state
        value_grid[i, j] = q_values.max().item()

# Plot Value Function Heatmap
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(value_grid, cmap='gist_gray')
plt.title('State-Value Function Heatmap')
plt.colorbar(label='State Value')
for i in range(grid_size):
    for j in range(grid_size):
        plt.text(j, i, f'{value_grid[i, j]:.2f}', ha='center', va='center', color='white')

# Plot Policy
plt.subplot(1, 2, 2)
plt.imshow(np.zeros_like(policy_grid), cmap='bone_r')
plt.title('Learned Policy')
for i in range(grid_size):
    for j in range(grid_size):
        plt.text(j, i, action_arrows.get(policy_grid[i, j], ' '), ha='center', va='center', color='black', fontsize=12)

plt.tight_layout()
plt.savefig(f"plots/{save_model_name}_{n_model}_value_grid.png")
plt.show()