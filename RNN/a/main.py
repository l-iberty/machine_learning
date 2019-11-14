from basic_rnn import *


def gen_data(size=1000000):
    """生成数据
    输入数据X：在时间t，Xt的值有50%的概率为，50%的概率为0:
    输出数据Y：在时间t，Yt的值有50%的概率为1，50%的概率为0. 除此之外，如果`Xt-3 == 1`，Yt为1的概率增加50%.
              如果`Xt-8 == 1`，则Yt为1的概率减少25%. 如果上述两个条件同时满足，则Yt为1的概率为75%.
    """
    X = np.array(np.random.choice(2, size=(size,)))
    Y = []
    for i in range(size):
        threshold = 0.5
        if X[i - 3] == 1:
            threshold += 0.5
        if X[i - 8] == 1:
            threshold -= 0.25
        if np.random.rand() > threshold:
            Y.append(0)
        else:
            Y.append(1)
    return X, np.array(Y)


def gen_batch(raw_data, num_batches, epoch_size):
    raw_x, raw_y = raw_data
    data_length = len(raw_x)

    # 将原始数据切分成num_batches个batch, 每个batch的长度为batch_size
    batch_size = data_length // num_batches
    data_x = np.zeros([num_batches, batch_size], dtype=np.int32)
    data_y = np.zeros([num_batches, batch_size], dtype=np.int32)
    for i in range(num_batches):
        data_x[i] = raw_x[i * batch_size:(i + 1) * batch_size]
        data_y[i] = raw_y[i * batch_size:(i + 1) * batch_size]

    # 把每个batch切分成num_steps个epoch, 每个epoch的长度为epoch_size
    num_steps = batch_size // epoch_size
    for i in range(num_steps):
        x = data_x[:, i * epoch_size:(i + 1) * epoch_size]
        y = data_y[:, i * epoch_size:(i + 1) * epoch_size]
        yield (x, y)


# gen_epoch是生成器的生成器, 依次生成n个gen_batch
def gen_epoch(n, num_batches, epoch_size):
    raw_data = gen_data()
    for i in range(n):
        yield gen_batch(raw_data, num_batches, epoch_size)


if __name__ == '__main__':
    plot_learning_curve(config.num_epoches, config.epoch_size, config.state_size)
