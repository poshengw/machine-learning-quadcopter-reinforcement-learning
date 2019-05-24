import tensorflow as tf
import numpy as np


""" Hyper Parameters"""
MEMORY_CAPACITY = 10000
TAU = 0.01 # soft replacement
GAMMA = 0.9 # reward discount
LR_C = 0.002 # learning rate for critic
LR_A = 0.001    # learning rate for actor
BATCH_SIZE = 64  # The size of the batch for memory

class DDPG(object):
    """ This is DDPG model and it contains the actor and critic."""
    def __init__(self, task):
        tf.reset_default_graph() # Rember to reset
        self.task = task
        s_dim = task.state_size
        a_dim = task.action_size
                
        # Set up the zeros space to initialize the memory space. Current state + next state + action + reward.
        # float 32 is a single precision (32 bit) floating point data type. While float 64 is a double precision (64 bit) floating data.  
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()
        
        # Define tf placeholder = (type, shpae, name)
        # None is an alias for numpy new axis. It creates an axis with length 1. 
        self.a_dim, self.s_dim = a_dim, s_dim,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        
        # Define the Actor with eval and target network
        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        
        # Define the Critic with eval and target network
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)
       
        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        
        # target net replacement. This means that how much percentage(tau) that the target net is going to learn
        # from eval net, and how many percentage (1-tau) to keep as whatever it was. 
        self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e)
                            for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]
        
        q_target = self.R + GAMMA * q_
        
        # in the feed_dic for the td_error, the self.a should change to action in memory
        # Train critic
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)
        
        # Train actor
        a_loss = - tf.reduce_mean(q) # the negative is for maximizing the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)
        
        self.sess.run(tf.global_variables_initializer())
        
    """ Based on state, choose the action a"""    
    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]
    
    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)
        
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim -1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]
        
        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})
    
    """ Store the experience into the memory"""
    def store_transition(self, s, a, r, s_):
        transition  = np.hstack((s, a, r, s_))
        index = self.pointer % MEMORY_CAPACITY # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1
        
    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            # Set up layers for the Actor training.
            net = tf.layers.dense(s, 32, activation=tf.nn.relu, name='l1', trainable=trainable)
            net = tf.layers.dense(net, 64, activation=tf.nn.relu, name='l2', trainable=trainable)
            net = tf.layers.dense(net, 32, activation=tf.nn.relu, name='l3', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.sigmoid, name='a', trainable=trainable)
            return a       
        
    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable) # Q(s, a)
        
    def reset_episode(self):
        state = self.task.reset()
        return state
        
        
 