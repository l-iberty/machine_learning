import tensorflow as tf
import numpy as np


class VariationalAutoencoder:
    def __init__(self, network_architecture, transfer_func=tf.nn.softplus, learning_rate=0.001, batch_size=100):
        self.network_architecture = network_architecture
        self.transfer_func = transfer_func
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # tf Graph input
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.network_architecture['n_input']])

        self._create_network()

        self._create_loss_optimizer()

        # Initializing tensorflow global variables
        init = tf.global_variables_initializer()

        # Launch the session
        self.sess = tf.InteractiveSession()
        self.sess.run(init)

    def __del__(self):
        print('VAE: close session')
        self.sess.close()

    def _create_network(self):
        # weights ans biases
        network_weights_biases = self._initialize_weights_biases(**self.network_architecture)

        # Use encoder network to determine mean and (log) variance of
        # Gaussian distribution in latent space.
        self.z_mean, self.z_log_sigma_square = \
            self._encoder_network(network_weights_biases['weights_encoder'], network_weights_biases['biases_encoder'])
        """ z_mean和z_log_sigma_square的计算公式参见 Auto-Encoding Variational Bayes - appendix C.2 """

        # Draw one sample z from Gaussian distribution
        n_z = self.network_architecture['n_z']
        epsilon = tf.random.normal(shape=[self.batch_size, n_z], mean=0, stddev=1, dtype=tf.float32)
        sigma = tf.sqrt(tf.exp(self.z_log_sigma_square))
        self.z = self.z_mean + tf.multiply(sigma, epsilon)  # z = mu + sigma * epsilon

        # Use decoder network to determine mean of Bernoulli distribution of reconstructed input.
        self.x_reconstr_mean = self._decoder_network(network_weights_biases['weights_decoder'],
                                                     network_weights_biases['biases_decoder'])

    def _create_loss_optimizer(self):
        generation_loss = -tf.reduce_sum(self.x * tf.log(self.x_reconstr_mean + 1e-10) +
                                         (1 - self.x) * tf.log(1 - self.x_reconstr_mean + 1e-10), axis=1)
        # Adding 1e-10 to avoid evaluation of log(0.0)
        # reduce_sum的参数: axis=0(对矩阵纵向求和) or 1(对矩阵横向求和)
        """ generation_loss的计算公式参见 Auto-Encoding Variational Bayes - appendix C.1 这里的x_reconstr_mean
        相当于公式中的y."""

        latent_loss = -0.5 * tf.reduce_sum(
            1 + self.z_log_sigma_square - tf.exp(self.z_log_sigma_square) - tf.square(self.z_mean), axis=1)
        """ latent_loss的计算公式参见 Auto-Encoding Variational Bayes - appendix B """

        self.loss = tf.reduce_mean(generation_loss + latent_loss)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def _initialize_weights_biases(self, n_hidden_encoder_1, n_hidden_encoder_2,
                                   n_hidden_decoder_1, n_hidden_decoder_2,
                                   n_input, n_z):
        initializer = tf.contrib.layers.xavier_initializer()
        all_weights_biases = dict()
        all_weights_biases['weights_encoder'] = {
            'h1': tf.Variable(initializer(shape=[n_input, n_hidden_encoder_1])),
            'h2': tf.Variable(initializer(shape=[n_hidden_encoder_1, n_hidden_encoder_2])),
            'out_mean': tf.Variable(initializer(shape=[n_hidden_encoder_2, n_z])),
            'out_log_sigma': tf.Variable(initializer(shape=[n_hidden_encoder_2, n_z]))
        }
        all_weights_biases['biases_encoder'] = {
            'b1': tf.Variable(tf.zeros(shape=[n_hidden_encoder_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros(shape=[n_hidden_encoder_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros(shape=[n_z], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros(shape=[n_z], dtype=tf.float32)),
        }
        all_weights_biases['weights_decoder'] = {
            'h1': tf.Variable(initializer(shape=[n_z, n_hidden_decoder_1])),
            'h2': tf.Variable(initializer(shape=[n_hidden_decoder_1, n_hidden_decoder_2])),
            'out_mean': tf.Variable(initializer(shape=[n_hidden_decoder_2, n_input])),
            'out_log_sigma': tf.Variable(initializer(shape=[n_hidden_decoder_2, n_input]))
        }
        all_weights_biases['biases_decoder'] = {
            'b1': tf.Variable(tf.zeros(shape=[n_hidden_decoder_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros(shape=[n_hidden_decoder_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros(shape=[n_input], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros(shape=[n_input], dtype=tf.float32)),
        }
        return all_weights_biases

    def _encoder_network(self, weights, biases):
        """ Generate probabilistic encoder (encoder network), which maps
        inputs onto a normal distribution in latent space.
        The transformation is parameterized and can be learned.
        """
        layer_1 = self.transfer_func(tf.matmul(self.x, weights['h1']) + biases['b1'])
        layer_2 = self.transfer_func(tf.matmul(layer_1, weights['h2']) + biases['b2'])
        z_mean = tf.matmul(layer_2, weights['out_mean']) + biases['out_mean']
        z_log_sigma_square = tf.matmul(layer_2, weights['out_log_sigma']) + biases['out_log_sigma']
        return z_mean, z_log_sigma_square

    def _decoder_network(self, weights, biases):
        """ Generate probabilistic decoder (decoder network), which maps points
        in latent space onto a Bernoulli distribution in data space.
        The transformation is parameterized and can be learned.
        """
        layer_1 = self.transfer_func(tf.matmul(self.z, weights['h1']) + biases['b1'])
        layer_2 = self.transfer_func(tf.matmul(layer_1, weights['h2']) + biases['b2'])
        x_reconstr_mean = tf.nn.sigmoid(tf.matmul(layer_2, weights['out_mean']) + biases['out_mean'])
        return x_reconstr_mean

    def fit(self, X):
        """ Train model based on input data."""
        optimizer, loss = self.sess.run([self.optimizer, self.loss], feed_dict={self.x: X})
        return loss

    def transform(self, X):
        """ Transform data by mapping it into latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution.
        return self.sess.run(self.z_mean, feed_dict={self.x: X})

    def generate(self, z_mu):
        """ Generate data by sampling from latent space."""
        # Note: This maps to mean of normal distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.x_reconstr_mean, feed_dict={self.z: z_mu})

    def reconstruct(self, X):
        """ Use VAE to reconstruct given data."""
        return self.sess.run(self.x_reconstr_mean, feed_dict={self.x: X})
