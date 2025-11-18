import gymnasium
import gym_gridworlds

# The `gym_gridworlds` import registers all the custom environments.
# We can then access the list of all registered environments from gymnasium.

all_envs = gymnasium.envs.registry.keys()

# Filter for the environments from this package
gridworld_envs = sorted([env_id for env_id in all_envs if env_id.startswith("Gym-Gridworlds/")])

print("Available Gym-Gridworlds environments:")
for env_id in gridworld_envs:
    print(f"- {env_id}")
