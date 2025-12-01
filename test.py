import gymnasium as gym
import gym_gridworlds
from stable_baselines3 import PPO, DQN

# vars
env_name = "TravelField-10x10-v0"
save_model_name = "travelfield"
distance_reward = False
no_stay = True
# start_pos = (0,0)
# random_goals = False

env = gym.make(f"Gym-Gridworlds/{env_name}", no_stay=no_stay, 
               distance_reward=distance_reward, 
            #    start_pos=start_pos, 
            #    random_goals=random_goals,
               render_mode="human")
model = DQN.load(f"test_envs/trained_models/{save_model_name}")

for episode in range(5):
    obs, info = env.reset()
    done = False
    episode_reward = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_reward += reward
    print(f"Episode {episode + 1}: Total Reward = {episode_reward}")

env.close() 