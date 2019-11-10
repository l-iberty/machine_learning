import tensorflow as tf
import numpy as np


class AutoEncoder:
    def __init__(self, m, n, learning_rate=0.01):
        """
        :param m: number of neurons in input/output layer
        :param n: number of neurons in hidden layer
        """
        self.m = m
        self.n = n
        self.learning_rate = learning_rate
        self.session = None

        """
        create the computation graph
        """

        # weights ans biases
        self.weight1 = tf.Variable(tf.random.normal(shape=[self.m, self.n]))
        self.weight2 = tf.Variable(tf.random.normal(shape=[self.n, self.m]))
        self.bias1 = tf.Variable(np.zeros(self.n).astype(np.float32))  # bias from hidden layer
        self.bias2 = tf.Variable(np.zeros(self.m).astype(np.float32))  # bias from output layer

        # placeholder for inputs
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.m])

        self.y = self.encoder(self.x)
        self.r = self.decoder(self.y)
        self.loss = tf.reduce_mean(tf.square(self.r - self.x))
        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def encoder(self, x):
        h = tf.matmul(x, self.weight1) + self.bias1
        return tf.nn.sigmoid(h)

    def decoder(self, y):
        h = tf.matmul(y, self.weight2) + self.bias2
        return tf.nn.sigmoid(h)

    def set_session(self, session):
        self.session = session

    def reduce_dimension(self, x):
        y = self.encoder(x)
        return self.session.run(y, feed_dict={self.x: x})

    def reconstruct(self, x):
        y = self.encoder(x)
        r = self.decoder(y)
        return self.session.run(r, feed_dict={self.x: x})

    def fit(self, X, epochs=100, batch_size=100):
        """
        :param X: 输入训练集
        :param epochs: 迭代次数, 过大会过拟合, 过小会欠拟合
        :param batch_size: 每次把多少数据放进神经网络进行训练
        :return:
        """
        N, D = X.shape  # N=训练集中的数据量 D=数据的维数
        num_batches = N // batch_size  # 全部数据集可以分为多少个批次
        loss = []
        for i in range(epochs):
            for j in range(num_batches):
                batch = X[j * batch_size:j * batch_size + batch_size]
                _, _loss = self.session.run([self.opt, self.loss], feed_dict={self.x: batch})
                if j % 100 == 0:
                    print('training epoch {0} batch {1} loss {2}'.format(i, j, _loss))
                loss.append(_loss)
        return loss
