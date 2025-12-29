#running this code lists all currently available environments in gym_gridworlds

import gymnasium
import gym_gridworlds

all_envs = gymnasium.envs.registry.keys()

gridworld_envs = sorted([env_id for env_id in all_envs if env_id.startswith("Gym-Gridworlds/")])

print("Available Gym-Gridworlds environments:")
for env_id in gridworld_envs:
    print(f"- {env_id}")
