"""
tensorflow矩阵基本操作 & 计算图可视化
"""
import tensorflow as tf

session = tf.InteractiveSession()

I_matrix = tf.eye(5)
print('5×5 identity matrix\n', I_matrix.eval())

X = tf.Variable(tf.eye(10))
X.initializer.run()
print('variable initialized to a 10×10 identity matrix X\n', X.eval())

A = tf.Variable(tf.random.normal([5, 10]))
A.initializer.run()
print('random 5×10 matrix A\n', A.eval())

product = tf.matmul(A, X)
print('AX\n', product.eval())

b_int = tf.Variable(tf.random.uniform(shape=[5, 10], minval=0, maxval=2, dtype=tf.int32))
b_int.initializer.run()
print('random 5×10 matrix of 0s and 1s b\n', b_int.eval())

b_float = tf.cast(b_int, dtype=tf.float32)
t_sum = tf.add(product, b_float)
t_sub = product - b_float
print('AX+b\n', t_sum.eval())
print('AX-b\n', t_sub.eval())

# 使用tensorboard将计算图可视化
tf.summary.FileWriter('summary_dir', session.graph)
# 激活conda-tensorflow环境后, 命令行输入:
# (tensorflow)$ tensorboard --logdir=path/to/summary_dir
# 打开浏览器输入 localhost:6006 即可访问

session.close()
