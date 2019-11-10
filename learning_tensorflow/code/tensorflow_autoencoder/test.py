import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from autoencoder import AutoEncoder

mnist = input_data.read_data_sets('MNIST_data')
train_data, train_labels = mnist.train.images, mnist.train.labels
test_data, test_labels = mnist.test.images, mnist.test.labels
'''
训练集和测试集分别包含55000和10000张图片, 每张图片表示为28×28矩阵, 矩阵元素是0-1浮点数, 越接近1则像素点颜色越接近黑色.
每个28×28矩阵被转换为长度为28×28=784的一维数组的形式存储.
'''

_, m = train_data.shape
autoEncoder = AutoEncoder(m, 256)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    autoEncoder.set_session(sess)
    loss = autoEncoder.fit(X=train_data, epochs=10)
    output = autoEncoder.reconstruct(train_data[0:100])
    # plot loss's change w.r.t epochs
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(loss)
    # plot original and reconstructed images
    n_rows, n_cols = 2, 8  # 2行, 一行原始图像, 一行重构图像
    idx = np.random.randint(0, 100, n_cols)
    # idx = np.array([i for i in range(n_cols)])
    figure, axes = plt.subplots(n_rows, n_cols, sharex=True, sharey=True, figsize=(10, 5))
    for fig, row in zip([train_data, output], axes):
        for i, ax in zip(idx, row):
            ax.imshow(fig[i].reshape(28, 28), cmap='Greys_r')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.show()
