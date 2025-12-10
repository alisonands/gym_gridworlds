import gymnasium as gym
import optuna 
import numpy as np
import gym_gridworlds

# ---------------------------------
# CREATE ENV (gymnasium)
# ---------------------------------

# ---------------------------------
# VARIABLES (change as required)
# ---------------------------------


def optimize_model(trial):
    alpha = trial.suggest_float("alpha", 0.001, 0.1)
    gamma = trial.suggest_float("gamma", 0.8, 0.99)
    eps = trial.suggest_float("eps", 0.001, 0.1)

    env = gym.make("Gym-Gridworlds/FourRooms-Original-13x13-v0", 
               no_stay=True, 
               random_goals=False, 
               start_pos = None,
               distance_reward=True)#, render_mode = "human")

    # ---- define observation and action spaces -----
    q = np.zeros((env.observation_space.n, env.action_space.n))


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

    env = gym.make("Gym-Gridworlds/FourRooms-Original-13x13-v0", 
                no_stay=True, 
                random_goals= False, 
                start_pos = None,
                distance_reward=True)
    
    # Evaluate the trained Q-table
    total_eval_reward = 0
    for _ in range(10):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = np.argmax(q[obs])
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        total_eval_reward += episode_reward
    
    mean_eval_reward = total_eval_reward / 10
    return mean_eval_reward

study = optuna.create_study(direction="maximize")
study.optimize(optimize_model, n_trials=20)

print("Number of finished trials:", len(study.trials))
print("Best trial:", study.best_trial.params)
    
