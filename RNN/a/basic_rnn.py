import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import config


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


"""以下是RNN模型, 代码参考https://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html"""

"""Placeholders
"""
x = tf.placeholder(tf.int32, [config.num_batches, config.epoch_size], name='input_placeholder')
y = tf.placeholder(tf.int32, [config.num_batches, config.epoch_size], name='labels_placeholder')
init_state = tf.zeros([config.num_batches, config.state_size])

"""RNN Inputs
"""
# Turn our x placeholder into a list of one-hot tensors
# The shape of rnn_inputs will be [num_batches, num_classes]
x_one_hot = tf.one_hot(x, config.num_classes)
rnn_inputs = tf.unstack(x_one_hot, axis=1)

"""Definition of rnn_cell
"""
with tf.variable_scope('rnn_cell'):
    W = tf.get_variable('W', [config.num_classes + config.state_size, config.state_size])
    b = tf.get_variable('b', [config.state_size], initializer=tf.constant_initializer(0.0))


def rnn_cell(rnn_input, state):
    with tf.variable_scope('rnn_cell', reuse=True):
        W = tf.get_variable('W', [config.num_classes + config.state_size, config.state_size])
        b = tf.get_variable('b', [config.state_size], initializer=tf.constant_initializer(0.0))
    return tf.tanh(tf.matmul(tf.concat([rnn_input, state], axis=1), W) + b)


"""Adding rnn_cells to graph
"""
state = init_state
rnn_outputs = []
for rnn_input in rnn_inputs:
    state = rnn_cell(rnn_input, state)
    rnn_outputs.append(state)
final_state = rnn_outputs[-1]

"""Predictions, loss, training step
"""
# logits and predictions
with tf.variable_scope('softmax'):
    W = tf.get_variable('W', [config.state_size, config.num_classes])
    b = tf.get_variable('b', [config.num_classes], initializer=tf.constant_initializer(0.0))
logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]
predictions = [tf.nn.softmax(logit) for logit in logits]

# Turn our y placeholder into a list of labels
y_as_list = tf.unstack(y, num=config.epoch_size, axis=1)

# losses and train_step
losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logit) for \
          logit, label in zip(logits, y_as_list)]
total_loss = tf.reduce_mean(losses)
train_step = tf.train.AdagradOptimizer(config.learning_rate).minimize(total_loss)


# Train the network
def train_network(num_epoches, epoch_size, state_size=config.state_size, verbose=True):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        training_losses = []
        for idx, epoch in enumerate(gen_epoch(num_epoches, config.num_batches, epoch_size)):
            training_loss = 0
            training_state = np.zeros((config.num_batches, state_size))
            if verbose:
                print("\nEPOCH", idx)
            for step, (X, Y) in enumerate(epoch):
                tr_losses, training_loss_, training_state, _ = \
                    sess.run([losses,
                              total_loss,
                              final_state,
                              train_step],
                             feed_dict={x: X, y: Y, init_state: training_state})
                training_loss += training_loss_
                if step % 100 == 0 and step > 0:
                    if verbose:
                        print("Average loss at step %d: %f" % (step, training_loss / 100))
                    training_losses.append(training_loss / 100)
                    training_loss = 0
    return training_losses


def plot_learning_curve(num_epoches, epoch_size, state_size):
    training_losses = train_network(num_epoches, epoch_size, state_size)
    plt.plot(training_losses)
    plt.show()
