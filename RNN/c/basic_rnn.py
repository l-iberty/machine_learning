import numpy as np
import tensorflow as tf


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

        self.x_one_hot = tf.one_hot(self.x, self.num_classes)  # Turn our x placeholder into a list of one-hot tensors
        self.rnn_inputs = tf.unstack(self.x_one_hot, axis=1)  # shape=[num_batches, num_classes]

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
            tf.contrib.rnn.static_rnn(cell, self.rnn_inputs, initial_state=self.init_state)

    def _create_loss_optimizer(self):
        with tf.variable_scope('softmax'):
            W = tf.get_variable('W', [self.state_size, self.num_classes])
            b = tf.get_variable('b', [self.num_classes], initializer=tf.constant_initializer(0.0))

        logits = [tf.matmul(rnn_output, W) + b for rnn_output in self.rnn_outputs]
        predictions = [tf.nn.softmax(logit) for logit in logits]

        # Turn our y placeholder into a list of labels
        y_as_list = tf.unstack(self.y, num=self.epoch_size, axis=1)

        losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logit)
                  for logit, label in zip(logits, y_as_list)]
        self.loss = tf.reduce_mean(losses)
        self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss)

    def fit(self, X, Y, init_state):
        loss, state, _ = self.sess.run([self.loss, self.final_state, self.optimizer],
                                       feed_dict={self.x: X, self.y: Y, self.init_state: init_state})
        return loss, state
