import gymnasium
import gym_gridworlds
import time
import pygame


env = gymnasium.make("Gym-Gridworlds/Penalty-Randomized-4x4-v0", render_mode="human")
obs, info = env.reset()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    env.render()
env.close()
