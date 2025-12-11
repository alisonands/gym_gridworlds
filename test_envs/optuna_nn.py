# ---------------------------------
# IMPORTS
# ---------------------------------
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
# custom environment
import gym_gridworlds
from gym_gridworlds.observation_wrappers import MatrixWithGoalWrapper, AddGoalWrapper
# optimization library
import optuna

# ---------------------------------
# OPTIMIZING FUNCTION
# ---------------------------------
def optimize_model(trial):
    # find best params for learning rate, discount factor and exploration/entropy
    learning_rate = trial.suggest_uniform("learning_rate", 1e-5, 1e-2)
    gamma = trial.suggest_uniform("gamma", 0.8, 0.99)

    # ---------- change here for ppo/dqn -------------
    # exploration_fraction = trial.suggest_uniform("exploration_fraction", 0, 0.3)
    ent_coef = trial.suggest_uniform("ent_coef", 0, 0.3)

    # ---------------------------------
    # CHANGE ENV HERE
    # ---------------------------------
    env_path = f"Gym-Gridworlds/TravelField-10x10-v0"

    env = gym.make(env_path,
                   no_stay = True,
                   start_pos = None,
                   random_goals = False,
                   distance_reward = True)
    
    env = MatrixWithGoalWrapper(env)

    model = PPO(
    "MlpPolicy",
    env,
    learning_rate=learning_rate,
    gamma=gamma,
    # exploration_fraction=exploration_fraction,
    ent_coef=ent_coef,
    verbose=1,
    )

    model.learn(total_timesteps = 40000)

    # evaluate over 10 eps and return mean reward
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes = 10)
    return mean_reward

# ---------------------------------
# OPTIMIZE FOR MAX REWARD VALUE
# ---------------------------------
study = optuna.create_study(direction="maximize")
study.optimize(optimize_model, n_trials=20, show_progress_bar=True)