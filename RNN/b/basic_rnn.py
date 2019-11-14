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
        self.rnn_outputs = []

        self._create_network()
        self._create_loss_optimizer()

        # Launch the session
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    def __del__(self):
        self.sess.close()

    def _initialize_weights_biases(self):
        self.weigts_biases = dict()
        self.weigts_biases['rnn_cell'] = {
            'W': tf.get_variable('W', [self.num_classes + self.state_size, self.state_size]),
            'bs': tf.get_variable('bs', [self.state_size], initializer=tf.constant_initializer(0.0))
        }
        self.weigts_biases['softmax'] = {
            'U': tf.get_variable('U', [self.state_size, self.num_classes]),
            'bp': tf.get_variable('bp', [self.num_classes], initializer=tf.constant_initializer(0.0))
        }

    def _create_network(self):
        self._initialize_weights_biases()

        # Adding rnn_cells to computation graph
        state = self.init_state
        for rnn_input in self.rnn_inputs:
            state = self._rnn_cell(rnn_input, state)
            self.rnn_outputs.append(state)
        self.final_state = self.rnn_outputs[-1]

    def _rnn_cell(self, rnn_input, state):
        W = self.weigts_biases['rnn_cell']['W']
        bs = self.weigts_biases['rnn_cell']['bs']
        return tf.tanh(tf.matmul(tf.concat([rnn_input, state], axis=1), W) + bs)

    def _create_loss_optimizer(self):
        U = self.weigts_biases['softmax']['U']
        bp = self.weigts_biases['softmax']['bp']
        logits = [tf.matmul(rnn_output, U) + bp for rnn_output in self.rnn_outputs]
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
