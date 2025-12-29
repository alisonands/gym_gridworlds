# open/display environment

import gymnasium
import gym_gridworlds
import time
import pygame

env_name = "Gym-Gridworlds/Penalty-Randomized-4x4-v0"
env = gymnasium.make(env_name, render_mode="human")
obs, info = env.reset()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    env.render()
env.close()