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
        self.fct_dims = fct_dims
        self.input_dims = input_dims
        self.sess = tf.Session()
        self.build_network()
        self.sess.run(tf.global_variables_initializer())
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
            # max pooling
            conv1_pooled = tf.layers.max_pooling2d(conv1_activated,
                                                    pool_size=(2,2), 
                                                    strides=1) # dim = 14x14x32
            conv2 = tf.layers.conv2d(inputs=conv1_activated, filters=64, 
                                     kernel_size=(3,3), strides=1, 
                                     name='conv2', 
                                     kernel_initializer=tf.variance_scaling_initializer(scale=2))
            conv3_activated = tf.nn.relu(conv2) # dim = 12x12x64
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

