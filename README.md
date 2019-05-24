# Machine Learning Engineer Nanodegree
# Reinforcement Learning
## Project: Train a Quadcopter How to Fly

### Project Summary:
- Implemented **reinforcement learning algorithm** to design an learning **agent** and **task** to fly a quadcopter.
- The **task** is to make quadcopter take off and reach the height of 100 from the starting point.
- Implemneted **Deep Deterministic Policy Gradients [(DDPG)](https://arxiv.org/abs/1509.02971)** for the agent.
- The **agent** contains **Actor and Critic**. Each of them contains **evaluation and targeting network** with **Adam optimizer**, and we control how much percent of the targeting network is going to learn from evaluation network. 
- **Actor**: <br/> Implemented 32, 64 and 32 nodes as hidden layers with relu as activation function. The output layer uses sigmoid function. The input is S (state) and the output is A (action).
- **Critic**: <br/> Implemented one layer with 30 nodes as hidden layer and it's using relu function as the activation function. The input is (S, A) and the output is Q value. 




Here are the parameters: Gamma reward discount 0.9. Tau = 0.01 for soft replacement. Critic learning rate: 0.002. Actor learning rate: 0.001. Memory capacity: 10000. Batch size: 64.



5. Policy to compute the action
7. Average reward obtained from each episode, it keeps tracking the best parameters found so far.  
8. to build in a mechanism to log/save the total rewards obtained in each episode to file. If the episode rewards are gradually increasing, this is an indication that your agent is learning.
9.  The reward function give the quadcopter a large amount of reward if it is closer to the target point. Othersie, the reward consistently provides a small amount of the reward.  


11. Most modern reinforcement learning algorithms benefit from using a replay memory or buffer to store and recall experience tuples. 
12. The project is about continuous state and action space. One popular choice is Deep Deterministic Policy Gradients or DDPG. It is actually an actor-critic method, but the key idea is that the underlying policy function used is deterministic in nature, with some noise added in externally to produce the desired stochasticity in actions taken.
13. The two main components of the algorithm, the actor and critic networks can be implemented using most modern deep learning libraries, such as Keras or TensorFlow.
14. Another thing to note is how the loss function is defined using action value (Q value) gradient. These gradients will need to be computed using the critic model, and fed in while training. Hence it is specified as part of the "inputs" used in the training function
15.  while the actor model is meant to map states to actions, the critic model needs to map (state, action) pairs to their Q-values. This is reflected in the input layers.
16. Critic: The final output of this model is the Q-value for any given (state, action) pair. However, we also need to compute the gradient of this Q-value with respect to the corresponding action vector, needed for training the actor model. 
17. We are now ready to put together the actor and policy models to build our DDPG agen


state space: In the sample task in task.py, we use the 6-dimensional pose of the quadcopter to construct the state of the environment at each timestep. Inspired by the methodology in the original DDPG paper, we make use of action repeats. For each timestep of the agent, we step the simulation action_repeats timesteps. If you are not familiar with action repeats, please read the Results section in the DDPG paper.
action space: The environment will always have a 4-dimensional action space, with one entry for each rotor (action_size=4). You can set the minimum (action_low) and maximum (action_high) values of each entry here. (rotor_speeds as actions?)

Policy
