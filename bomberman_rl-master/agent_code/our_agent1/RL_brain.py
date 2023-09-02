import numpy as np
import os
import tensorflow as tf
import settings as s

class DeepQNetwork(object):
    def __init__(self, lr, n_actions, name, fct_dims, 
                 input_dims = (s.ROWS, s.COLS)) -> None:
        self.lr = lr
        self.n_actions = n_actions
        self.name = name
        self.fct_dims = fct_dims # number of neurons in the fully connected layer
        self.input_dims = input_dims # input dimensions (17 * 17)
        self.sess = tf.Session() # tensorflow session
        self.build_network() # build the network
        self.sess.run(tf.global_variables_initializer()) # initialize the variables
        self.saver = tf.train.Saver()
        self.checkpoint_file = os.path.join('deepqnet.ckpt')
        self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                        scope=self.name)
        
    def build_network(self):
        with tf.variable_scope(self.name):
            self.input = tf.placeholder(tf.float32, shape=[None, 
                                                           *self.input_dims], 
                                        name='inputs')
            self.actions = tf.placeholder(tf.float32, shape=[None, 
                                                             self.n_actions], 
                                          name='action_taken')
            conv1 = tf.layers.conv2d(inputs=self.input, filters=32, 
                                     kernel_size=(3,3), strides=1, 
                                     name='conv1', 
                                     kernel_initializer=tf.variance_scaling_initializer(scale=2))
            conv1_activated = tf.nn.relu(conv1) # dim = 15x15x32
            conv2 = tf.layers.conv2d(inputs=conv1_activated, filters=64, 
                                     kernel_size=(3,3), strides=1, 
                                     name='conv2', 
                                     kernel_initializer=tf.variance_scaling_initializer(scale=2))
            conv3_activated = tf.nn.relu(conv2) # dim = 13x13x64
            flat = tf.layers.flatten(conv3_activated)
            dense1 = tf.layers.dense(flat, units=128, activation=tf.nn.relu, 
                                     kernel_initializer=tf.variance_scaling_initializer(scale=2))
            self.Q_values = tf.layers.dense(dense1, units=self.n_actions, 
                                            kernel_initializer=tf.variance_scaling_initializer(scale=2))
            
            # build q_target
            self.q_target = tf.placeholder(tf.float32, shape=[None, 
                                                              self.n_actions], 
                                           name='q_value')
            # build loss function
            self.loss = tf.reduce_mean(tf.square(self.Q_values - self.q_target))
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)



    def save_model(self):
        print('... saving model ...')
        self.saver.save(self.sess, self.checkpoint_file)

    def load_model(self):
        print('... loading model ...')
        self.saver.restore(self.sess, self.checkpoint_file)

    class Agent(object):
        def __init__(self, alpha, gamma, mem_size, n_actions, epsilon,
                     batch_size, replace_target=1000, input_dims=(s.ROWS, s.COLS), 
                     q_next_dir='tmp/q_next', q_eval_dir='tmp/q_eval'):
            self.action_space = [i for i in range(n_actions)]
            self.gamma = gamma
            self.mem_size = mem_size
            self.mem_cntr = 0
            self.epsilon = epsilon
            self.batch_size = batch_size
            self.replace_target = replace_target
            self.q_next = DeepQNetwork(alpha, n_actions, input_dims=input_dims, 
                                        name='q_next', fct_dims=512)
            self.q_eval = DeepQNetwork(alpha, n_actions, input_dims=input_dims, 
                                        name='q_eval', fct_dims=512)
            self.state_memory = np.zeros((self.mem_size, *input_dims))
            self.new_state_memory = np.zeros((self.mem_size, *input_dims))
            self.action_memory = np.zeros((self.mem_size, n_actions), dtype=np.int8)
            self.reward_memory = np.zeros(self.mem_size)
            self.terminal_memory = np.zeros(self.mem_size, dtype=np.int8)
            self.learn_step_counter = 0 # count how many times we learn
            self.q_next_dir = q_next_dir
            self.q_eval_dir = q_eval_dir
            self.replace_target_cnt = 0
        
        def store_transition(self, state, action, reward, state_, terminal):
            index = self.mem_cntr % self.mem_size
            self.state_memory[index] = state
            actions = np.zeros(self.n_actions)
            actions[action] = 1.0
            self.action_memory[index] = actions
            self.reward_memory[index] = reward
            self.new_state_memory[index] = state_
            self.terminal_memory[index] = 1 - terminal
            self.mem_cntr += 1

        def choose_action(self, observation):
            rand = np.random.random()
            if rand < self.epsilon:
                action = np.random.choice(self.action_space)
            else:
                actions = self.q_eval.sess.run(self.q_eval.Q_values, 
                                                feed_dict={self.q_eval.input:observation})
                action = np.argmax(actions)
            return action
        
        def learn(self):
            if self.mem_cntr % self.replace_target == 0:
                self.update_graph()
            
            max_mem = self.mem_cntr if self.mem_cntr < self.mem_size else self.mem_size
            batch = np.random.choice(max_mem, self.batch_size) # experience replay

            state_batch = self.state_memory[batch]
            action_batch = self.action_memory[batch]
            action_values = np.array(self.action_space, dtype=np.int8)
            action_indices = np.dot(action_batch, action_values)
            reward_batch = self.reward_memory[batch]
            new_state_batch = self.new_state_memory[batch]
            terminal_batch = self.terminal_memory[batch]

            q_eval = self.q_eval.sess.run(self.q_eval.Q_values, 
                                            feed_dict={self.q_eval.input:new_state_batch})
            q_next = self.q_next.sess.run(self.q_next.Q_values, 
                                            feed_dict={self.q_next.input:new_state_batch})
            q_target = q_eval.copy()
            batch_index = np.arange(self.batch_size, dtype=np.int32)
            q_target[batch_index, action_indices] = reward_batch + \
                                                    self.gamma*np.max(q_next, axis=1)*terminal_batch

            _ = self.q_eval.sess.run(self.q_eval.train_op, 
                                        feed_dict={self.q_eval.input:state_batch, 
                                                    self.q_eval.actions:q_target})
            self.epsilon = self.epsilon * (1-1e-7) if self.epsilon > \
                            0.01 else 0.01
            self.learn_step_counter += 1
        
        def save_models(self):
            self.q_eval.save_model()
            self.q_next.save_model()

        def load_models(self):
            self.q_eval.load_model()
            self.q_next.load_model()

        def update_graph(self):
            t_params = self.q_next.params
            e_params = self.q_eval.params
            for t, e in zip(t_params, e_params):
                self.q_eval.sess.run(tf.assign(t, e))
            self.replace_target_cnt += 1
            if self.replace_target_cnt % 10 == 0:
                print('... target network updated ...')
