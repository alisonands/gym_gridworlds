from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import gym_gridworlds
from gym_gridworlds.observation_wrappers import MatrixWithGoalWrapper


# vars
env_name = "TravelField-10x10-v0"
save_model_name = "travelfield"
n_model = "PPO"
eval_model_name = f"{save_model_name}_eval_{n_model}"
no_stay = True
distance_reward = True
start_pos = None
random_goals = False
LOG_DIR = "log_dir/"

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

env_name = "TravelField-10x10-v0"

eval_env = gym.make(f"Gym-Gridworlds/{env_name}", 
                    no_stay=no_stay, 
                    distance_reward=distance_reward,
                    start_pos = start_pos,
                    random_goals = random_goals,
                    )
# wrap envs
eval_env = MatrixWithGoalWrapper(eval_env)
eval_env = Monitor(eval_env, f"{LOG_DIR}/{eval_model_name}")

trained_model = PPO.load(f"trained_models/{save_model_name}_{n_model}")
mean_reward, std_reward = evaluate_policy(trained_model, eval_env, n_eval_episodes=10)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")