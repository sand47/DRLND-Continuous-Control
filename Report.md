## DRL - DDPG Algorithm - Reacher Continuous Control

### Model Architecture
This project utilised the DDPG (Deep Deterministic Policy Gradient) architecture outlined in the [Udacity](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-bipedal)

## State and Action Spaces
In this environment, when a double-jointed arm can move to target locations a reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector must be a number between -1 and 1.

## Learning Algorithm

The agent training utilised the `ddpg` function in the DDPG_Continuous_Control notebook.

It continues episodical training via the the ddpg agent until `n_episoses` is reached or until the environment tis solved. The  environment is considered solved when the average reward (over the last 100 episodes) is at least +30.0. Note if the number of agents is >1 then the average reward of all agents at that step is used.

Each episode continues until `max_t` time-steps is reached or until the environment says it's done.

As above, a reward of +0.1 is provided for each step that the agent's hand is in the goal location.

The DDPG agent is contained in [`ddpg_agent.py`](https://github.com/sand47/DRLND-Continuous-Control/blob/master/ddpg_agent.py)

### DDPG Hyper Parameters
- n_episodes (int): maximum number of training episodes
- max_t (int): maximum number of timesteps per episode
- num_agents: number of agents in the environment

Where
`n_episodes=300`, `max_t=1000`

### DDPG Agent Hyper Parameters

- BUFFER_SIZE = int(1e6)  # replay buffer size
- BATCH_SIZE = 128        # minibatch size
-GAMMA = 0.99            # discount factor
-TAU = 1e-3              # for soft update of target parameters
-LR_ACTOR = 1e-4        # learning rate of the actor
-LR_CRITIC = 1e-3        # learning rate of the critic
-WEIGHT_DECAY = 0        # L2 weight decay
- LEARN_EVERY = 20        # learning timestep interval
- LEARN_NUM = 10          # number of learning passes
- OU_SIGMA = 0.2          # Ornstein-Uhlenbeck noise parameter
- OU_THETA = 0.15         # Ornstein-Uhlenbeck noise parameter
- EPSILON = 1.0           # explore->exploit noise process added to act step
- EPSILON_DECAY = 1e-6    # decay rate for noise process

### Neural Networks

Actor and Critic network models were defined in [`ddpg_model.py`](https://github.com/hortovanyi/DRLND-Continuous-Control/blob/master/ddpg_model.py).

The Actor networks utilised two fully connected layers with 256 and 128 units with relu activation and tanh activation for the action space. The network has an initial dimension the same as the state size.

The Critic networks utilised two fully connected layers with 256 and 128 units with leaky_relu activation. The critic network has  an initial dimension the size of the state size plus action size.

## Plot of rewards
![Reward Plot](https://github.com/sand47/DRLND-Continuous-Control/blob/master/images/reward%20plot.PNG)

```
Episode 100 (223 sec)  -- 	Min: 36.1	Max: 39.6	Mean: 38.9	Mov. Avg: 27.8
Episode 101 (223 sec)  -- 	Min: 36.7	Max: 39.6	Mean: 38.7	Mov. Avg: 28.2
Episode 102 (222 sec)  -- 	Min: 34.6	Max: 39.7	Mean: 38.2	Mov. Avg: 28.6
Episode 103 (224 sec)  -- 	Min: 35.1	Max: 39.5	Mean: 38.0	Mov. Avg: 28.9
Episode 104 (224 sec)  -- 	Min: 24.2	Max: 39.5	Mean: 37.0	Mov. Avg: 29.3
Episode 105 (225 sec)  -- 	Min: 37.1	Max: 39.6	Mean: 39.1	Mov. Avg: 29.6
Episode 106 (222 sec)  -- 	Min: 37.1	Max: 39.5	Mean: 38.3	Mov. Avg: 30.0
Episode 107 (220 sec)  -- 	Min: 32.5	Max: 39.5	Mean: 38.0	Mov. Avg: 30.3

Environment SOLVED in 7 episode	Moving Average =30.3 over last 100 episodes
```
## Ideas for Future Work

Algorithms like PPO, A3C, and D4PG that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience can be tested out and I first tried 0.3 OUNoise but did not get a better output so I reduced it to 0.2. So playing around with hyperparametrs might result in better architecture and 
also we can try chaning the number of conv layer as well
