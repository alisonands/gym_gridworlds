from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import gym_gridworlds


# vars
env_name = "FourRooms-Original-13x13-v0"
save_model_name = "four_rooms"
n_model = "DQN"
eval_model_name = f"{save_model_name}_eval_{n_model}"
no_stay = True
distance_reward = True
start_pos = None
random_goals = False
LOG_DIR = "log_dir/"

# wrapper class
class OneHotWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        n = env.observation_space.n
        self.observation_space = gym.spaces.Box(0, 1, (n,), dtype=np.float32)

    def observation(self, obs):
        v = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        v[obs] = 1.0
        return v

# Visualize the learned policy and value function
trained_model = DQN.load(f"trained_models/{save_model_name}_{n_model}")
grid_size = 13
policy_grid = np.zeros((grid_size, grid_size), dtype=int)
value_grid = np.zeros((grid_size, grid_size))

# Action mapping for arrows
action_arrows = {0: '←', 1: '↓', 2: '→', 3: '↑'}

def one_hot(obs, size):
    v = np.zeros(size, dtype=np.float32)
    v[obs] = 1.0
    return v

for i in range(grid_size):
    for j in range(grid_size):
        obs = i * grid_size + j
        one_hot_obs = one_hot(obs, grid_size*grid_size)
        action, _ = trained_model.predict(one_hot_obs, deterministic=True)
        policy_grid[i, j] = action
        
        # For DQN, the state-value is the max of the Q-values for that state
        obs_tensor, _ = trained_model.policy.obs_to_tensor(np.array([one_hot_obs]))
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
plt.show()