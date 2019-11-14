import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn


class RNN_Model:
    def __init__(self, num_batches, num_classes, epoch_size, state_size, learning_rate):
        self.num_batches = num_batches
        self.num_classes = num_classes
        self.epoch_size = epoch_size
        self.state_size = state_size
        self.learning_rate = learning_rate

        tf.reset_default_graph()

        self.x = tf.placeholder(tf.int32, [self.num_batches, self.epoch_size], name='input_placeholder')
        self.y = tf.placeholder(tf.int32, [self.num_batches, self.epoch_size], name='labels_placeholder')
        self.init_state = tf.zeros([self.num_batches, self.state_size])

        self.rnn_inputs = tf.one_hot(self.x, self.num_classes) # shape=(num_batches, epoch_size, num_classes)

        self._create_network()
        self._create_loss_optimizer()

        # Launch the session
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    def __del__(self):
        self.sess.close()

    def _create_network(self):
        cell = tf.contrib.rnn.BasicRNNCell(self.state_size)
        self.rnn_outputs, self.final_state = \
            dynamic_rnn(cell, self.rnn_inputs, initial_state=self.init_state)
        # shape of rnn_outputs is (num_batches, epoch_size, state_size)
        # shape of final_state is (num_batches, state_size), the same as before

    def _create_loss_optimizer(self):
        with tf.variable_scope('softmax'):
            W = tf.get_variable('W', [self.state_size, self.num_classes])
            b = tf.get_variable('b', [self.num_classes], initializer=tf.constant_initializer(0.0))

        logits = tf.reshape(
            tf.matmul(tf.reshape(self.rnn_outputs, [-1, self.state_size]), W) + b, # '-1'表示这一维的大小由函数自动计算
            [self.num_batches, self.epoch_size, self.num_classes])
        predictions = tf.nn.softmax(logits)

        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=logits)
        self.loss = tf.reduce_mean(losses)
        self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss)

    def fit(self, X, Y, init_state):
        loss, state, _ = self.sess.run([self.loss, self.final_state, self.optimizer],
                                       feed_dict={self.x: X, self.y: Y, self.init_state: init_state})
        return loss, state
