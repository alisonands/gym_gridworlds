## Overview
The overall objective of our project is to design, implement, and evaluate a learning-based navigation policy that can guide an autonomous agent through 2D obstacle environments. 

We will build
gridworld environments to train and compare several RL agents (Q-learning, DQN, and PPO). 

We evaluate them based on reward functions and analyze how reward
shaping and hyperparameter choices influence stability, sample efficiency, and learned behavior.

A summary of the project can be found here: [final_report.pdf](https://alisonands.github.io/docs/505_final_report_v2.pdf)

## Structure
All code for training and testing are in the [test_envs](test_envs) folder. Each file trains, tests, and evaluates models for the respective environment.

## Results
- 3x3 trained on DQN (more details in report) <br>
<img src="results/DQN3x3.gif" alt="3x3DQN" width="50%">
- 3x3 trained on PPO <br>
<img src="results/3x3PPO.gif" alt="3x3PPO" width="50%">
- Four Rooms env <br>
<img src="results/four_rooms.gif" alt="four_rooms" width="50%">
- Travel Field env <br>
<img src="results/travel_field.gif" alt="travel_field" width="50%">
- Bigger Travel Field (Unfortunately, this hasen't worked out. yet...) <br>
<img src="results/travel_field_big.gif" alt="travel_field_big" width="50%">