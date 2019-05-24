# Machine Learning Engineer Nanodegree
# Reinforcement Learning
## Project: Train a Quadcopter How to Fly

### Problem Statement
1. Design an agent to fly a quadcopter, and then train it using a reinforcement learning algorithm.
2. Define the task and design a reinforcement learning agent.
3. The Task
4. The Agent
5. Policy to compute the action
6. Deep Deterministic Policy Gradients (DDPG) 
7. Average reward obtained from each episode, it keeps tracking the best parameters found so far.  
8. to build in a mechanism to log/save the total rewards obtained in each episode to file. If the episode rewards are gradually increasing, this is an indication that your agent is learning.

9. This is a taking off task for quadcopter. The goal is to reach the height of 100 from the starting point
10. Initial the DDPG object that contains the Actor and Critic

11. Most modern reinforcement learning algorithms benefit from using a replay memory or buffer to store and recall experience tuples. 
12. The project is about continuous state and action space. One popular choice is Deep Deterministic Policy Gradients or DDPG. It is actually an actor-critic method, but the key idea is that the underlying policy function used is deterministic in nature, with some noise added in externally to produce the desired stochasticity in actions taken.
13. The two main components of the algorithm, the actor and critic networks can be implemented using most modern deep learning libraries, such as Keras or TensorFlow.
14. Another thing to note is how the loss function is defined using action value (Q value) gradient. These gradients will need to be computed using the critic model, and fed in while training. Hence it is specified as part of the "inputs" used in the training function
15.  while the actor model is meant to map states to actions, the critic model needs to map (state, action) pairs to their Q-values. This is reflected in the input layers.
16. Critic: The final output of this model is the Q-value for any given (state, action) pair. However, we also need to compute the gradient of this Q-value with respect to the corresponding action vector, needed for training the actor model. 
17. We are now ready to put together the actor and policy models to build our DDPG agen
