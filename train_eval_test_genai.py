import gymnasium as gym
import gym_gridworlds
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

# 1. Create the Gridworld environment
# You can try other grids like "3x5_two_room_quicksand" or "6x6_danger_maze"
env = gym.make("Gym-Gridworlds/Full-4x5-v0", grid="4x4_quicksand")#, render_mode="human")

# 2. Instantiate the DQN model
# "MlpPolicy" means the agent will use a Multi-Layer Perceptron (a standard neural network)
# to decide on actions. This works well for discrete observation spaces.
# The verbose=1 argument will print training progress.
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=5e-4,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=32,
    tau=1.0,
    gamma=0.85,
    train_freq=(1, "step"),
    gradient_steps=1,
    target_update_interval=250,
    exploration_fraction=0.1,
    exploration_final_eps=0.05,
    verbose=1,
)

# 3. Train the agent
# The agent will learn for 50,000 timesteps.
# For more complex environments, you will need more steps.
print("--- Starting Training ---")
model.learn(total_timesteps=50000, progress_bar=False)
print("--- Training Finished ---")

# 4. Save the trained model
model.save("dqn_gridworld_quicksand")

# --- Evaluation ---
# To see how well it learned, let's load the model and evaluate it.
# We need to wrap the environment to record video for rendering during evaluation.
eval_env = gym.make("Gym-Gridworlds/Full-4x5-v0", grid="4x4_quicksand", render_mode="human")

# Load the trained agent
trained_model = DQN.load("dqn_gridworld_quicksand", env=eval_env)

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(trained_model, eval_env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Watch the trained agent play for 5 episodes
print("\n--- Watching the trained agent ---")
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

eval_env.close()
