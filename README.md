# Machine Learning Engineer Nanodegree
# Reinforcement Learning
## Project: Train a Quadcopter How to Fly

### Project Summary:
- Implemented **reinforcement learning algorithm** to design an learning **agent** and **task** to fly a quadcopter.
- The **task** is to make quadcopter take off and reach the height of 100 from the starting point, which is **continuous state and action space**. 
- **State space**: 6-dimensional pose of the quadcopter to construct the state of the environment at each timestep.
**Action space**: 4-dimensional action space with one entry for each rotor.
- Implemneted **actor-critic** method with **Deep Deterministic Policy Gradients [(DDPG)](https://arxiv.org/abs/1509.02971)** for the agent. The key idea is that the underlying **policy function** used is deterministic in nature, with some noise added in externally to produce the desired stochasticity in actions taken.
- The **agent** contains **Actor and Critic**. Each of them contains **evaluation and targeting network** with **Adam optimizer** and implemneted by **Tensorflow**. The percentage of the targeting network is going to learn from evaluation network is pre-defined as parameter. 
- **Actor**: <br/> Implemented 32, 64 and 32 nodes as hidden layers with relu as activation function. The output layer uses sigmoid function. The input is S (state) and the output is A (action).
- **Critic**: <br/> Implemented one layer with 30 nodes as hidden layer and it's using relu function as the activation function. The input is (S, A) and the output is Q value. 
- While the actor model is meant to map states to actions, the critic model needs to map (state, action) pairs to their **Q-values**. This is reflected in the input layers. We also need to compute the gradient of this Q-value with respect to the corresponding action vector, which is needed for training the actor model. 
- A large amount of reward is given by **reward function** if quadcopter is closer to the target point. Othersie, the reward consistently provides a small amount of the reward.  
- Averaged **reward** obtained from each episode and kept tracking the best parameters found.  
- Applied **replay memory** or **buffer** to store and recall experience for the reinforcement learning algorithm.
