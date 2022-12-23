# Maze Navigation using Reinforcement Learning #
This repository contains the code and report for a reinforcement learning project to navigate a maze environment using Dynamic Programming, Monte-Carlo learning, and Temporal-Difference learning. This project was completed as part of the Reinforcement Learning module at Imperial College London, attaining a distinction!

## Getting Started ##
To run the code in this repository, you will need to have the following dependencies installed:
  - Numpy
  - Matplotlib
  
You can install these dependencies by running the following command:
``` linux
pip install numpy matplotlib
```

You can then clone this repository and navigate to the local directory:
``` linux
git clone https://github.com/alexmkv01/Solving-Maze-Environment.git
cd Solving-Maze-Environment
```

## Environment ##

<p align="center">
  <img width="200" alt="grid_world" src="https://user-images.githubusercontent.com/72558653/209362584-9ab650f7-9414-4173-ab6c-808f0a12912d.png">
</p>

The maze environment is modeled as a Markov Decision Process (MDP) and is represented in a grid where black squares represent obstacles, and dark-grey squares represent absorbing states with specific rewards. Absorbing states are terminal states, and there are no transitions from an absorbing state to any other state.
The agent can choose from four actions at each time step:
  - a<sub>0</sub> = going north of its current state
  - a<sub>1</sub> = going east of its current state
  - a<sub>2</sub> = going south of its current state
  - a<sub>3</sub> = going west of its current state
  
Each action has a probability p (set to 0.95) of success, leading to the expected direction. If the action fails, there is an equal probability of the agent going in any other direction. If the action leads the agent to a wall or obstacle, it stays in its current state. Each action performed by the agent in the environment gives a reward of -1. The reward state R<sub>0</sub> gives a reward of 500 (goal state) and all the other absorbing states give a reward of −50 (penalty states). Any action performed by the agent in the environment also gives a reward of −1. 

## Agent Classes ## 
We have implemented three agent classes to navigate the maze, with the following strategies.

  - **`DP_agent`**: A dynamic programming agent to solve the maze. 
    - Value iteration 
    - Policy Iteration 
  - **`MC_agent`**: A Monte-Carlo learning agent to solve the maze.
    - First Visit MC
    - Every Visit MC
  - **`TD_agent`**: A temporal-difference learning agent to solve the maze.
    - SARSA on-policy
    - Q-learning off-policy

## Running the Code ##
To run the code, simply open and run the python file. The python file also includes code for generating the various plots used for the report.

## Results ## 
The results of our experiments can be found in the report included in this repository. We were able to successfully navigate the maze using Dynamic Programming, Monte-Carlo learning, and Temporal-Difference learning. The optimal policy and value grid is given below.

<p align="center">
  <img width="200" alt="DP Optimal Values" src="https://user-images.githubusercontent.com/72558653/209364099-f33f711a-2ad5-4661-9adc-d0b17b1635a7.png">
  <img width="200" alt="DP_Optimal_Policy" src="https://user-images.githubusercontent.com/72558653/209364117-e5327df0-0bfc-415a-914a-cb4431bd3600.png">
</p>

## Report ## 
A report detailing our implementation, analysing the effect of changing various hyperparameters, comparing the various learners and outlining the final results can be found in the "coursework1_report.pdf" file included in this repository. 



