import matplotlib.pyplot as plt
import numpy as np
from variational_autoencoder import VariationalAutoencoder
from tensorflow.examples.tutorials.mnist import input_data


def train(X, network_architecture, learning_rate=0.001, batch_size=100, train_epochs=10, display_step=5):
    N, D = X.shape
    num_batches = N // batch_size
    vae = VariationalAutoencoder(network_architecture, learning_rate=learning_rate, batch_size=batch_size)
    for epoch in range(train_epochs):
        avg_loss = 0.0
        for i in range(num_batches):
            batch = X[i * batch_size:i * batch_size + batch_size]
            loss = vae.fit(batch)
            avg_loss += loss / N * batch_size
        if epoch % display_step == 0:
            print('epoch: %d loss= %.5f' % (epoch, avg_loss))
    return vae


def illustrate_reconstruction_quality(mnist):
    """ Based on trained VAE we can sample some test inputs and visualize how well the VAE can
    reconstruct those. In general the VAE does really well."""
    train_data = mnist.train.images
    test_data = mnist.test.images

    network_architecture = {
        'n_hidden_encoder_1': 500,  # 1st layer encoder neurons
        'n_hidden_encoder_2': 500,  # 2nd layer encoder neurons
        'n_hidden_decoder_1': 500,  # 1st layer decoder neurons
        'n_hidden_decoder_2': 500,  # 2nd layer decoder neurons
        'n_input': 28 * 28,  # MNIST data input (img shape: 28*28)
        'n_z': 20}  # dimensionality of latent space 该值越大, 重构图像越清晰

    vae = train(X=train_data, network_architecture=network_architecture, train_epochs=10)

    x_input = test_data[0:100]
    x_reconstruct = vae.reconstruct(x_input)

    # plotting reconstructions
    plt.figure(figsize=(6, 10))
    n_rows, n_cols = 5, 2
    for i in range(n_rows):
        plt.subplot(n_rows, n_cols, 2 * i + 1)  # 在一个n_rows行n_cols列的网格中, 将图像画在位置(2i+1)处.
        """ 对于一个3×2网格, plt.subplot定义的网格位置编号为:
        +-+-+
        |1|2|
        +-+-+
        |3|4|
        +-+-+
        |5|6|
        +-+-+
        """
        plt.imshow(x_input[i].reshape(28, 28), vmin=0, vmax=1, cmap='Greys_r')
        plt.title('Test input')
        plt.colorbar()  # Add a colorbar to a plot.
        plt.subplot(n_rows, n_cols, 2 * i + 2)
        plt.imshow(x_reconstruct[i].reshape(28, 28), vmin=0, vmax=1, cmap='Greys_r')
        plt.title('Reconstruction')
        plt.colorbar()
        plt.tight_layout()  # adjusts subplot params so that the subplot(s) fits in to the figure area.
    plt.show()


def illustrate_latent_space(mnist):
    """ We train a VAE with 2-d latent space and illustrate how the encoder encodes some of the labeled
    inputs(collapsing the Gaussian distribution in the latent space to its mean. 把隐空间中的正态分布折叠
    到它的均值上). This gives us some insights into the structure of latent space. """
    train_data = mnist.train.images
    test_data = mnist.test.images

    network_architecture = {
        'n_hidden_encoder_1': 500,  # 1st layer encoder neurons
        'n_hidden_encoder_2': 500,  # 2nd layer encoder neurons
        'n_hidden_decoder_1': 500,  # 1st layer decoder neurons
        'n_hidden_decoder_2': 500,  # 2nd layer decoder neurons
        'n_input': 28 * 28,  # MNIST data input (img shape: 28*28)
        'n_z': 2}  # dimensionality of latent space 该值越大, 重构图像越清晰

    vae_2d = train(X=train_data, network_architecture=network_architecture, train_epochs=10)

    nx = ny = 20
    x_values = np.linspace(-3, 3, nx)
    y_values = np.linspace(-3, 3, ny)
    plt.figure(figsize=(8, 10))
    canvas = np.zeros(shape=(28 * nx, 28 * ny))  # 以一个28*28的点阵为一个图像单元, 一共nx*ny个图像单元
    for i, xi in enumerate(x_values):
        for j, yj in enumerate(y_values):
            z_mu = np.array([[xi, yj]] * vae_2d.batch_size)
            x_mean = vae_2d.generate(z_mu)  # REQUIRE: n_z == 2
            canvas[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = x_mean[0].reshape(28, 28)
            # 从左上角开始从左至右逐行绘图, 每次绘制一个28×28的图像
            # plt.imshow(canvas, origin='upper', cmap='Greys_r')  # 查看每次生成的图像
    plt.imshow(canvas, origin='upper', cmap='Greys_r')
    plt.tight_layout()
    plt.show()


def illustrate_latent_space2(mnist):
    """ Another way of getting insights into the latent space is to use the decoder network
    to plot reconstructions at the positions in the latent space for which they have been generated. """
    train_data = mnist.train.images
    test_data, test_labels = mnist.test.images, mnist.test.labels
    # images是手写数字的图片; labels是每个图片的标签, 即每张图片对应的数字

    network_architecture = {
        'n_hidden_encoder_1': 500,  # 1st layer encoder neurons
        'n_hidden_encoder_2': 500,  # 2nd layer encoder neurons
        'n_hidden_decoder_1': 500,  # 1st layer decoder neurons
        'n_hidden_decoder_2': 500,  # 2nd layer decoder neurons
        'n_input': 28 * 28,  # MNIST data input (img shape: 28*28)
        'n_z': 2}  # dimensionality of latent space 该值越大, 重构图像越清晰

    vae_2d = train(X=train_data, network_architecture=network_architecture, train_epochs=10)

    X, labels = test_data[0:5000], test_labels[0:5000]
    z_mu = vae_2d.transform(X)
    plt.figure(figsize=(8, 6))
    plt.scatter(x=z_mu[:, 0], y=z_mu[:, 1], c=labels)  # c: 颜色序列, 每个数字代表一种颜色
    # x[a:b, c:d] 取a~(b-1)行和c~(d-1)列范围内的子矩阵
    # [:, a] 取第a列, 并以行向量的形式返回. 如果要以列向量形式返回则需要[:, a:a+1]
    plt.colorbar()
    plt.grid()
    plt.show()


def main():
    mnist = input_data.read_data_sets('MNIST_data')
    illustrate_reconstruction_quality(mnist)
    illustrate_latent_space(mnist)
    illustrate_latent_space2(mnist)


if __name__ == '__main__':
    main()
