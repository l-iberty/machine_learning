# Global config variables
num_batches = 200
num_classes = 2
learning_rate = 0.1

# epoch_size=5, 只能学习到第一条依赖X_{t-3}, 不能学习到X_{t-8}. 交叉熵接近0.52
#num_epoches = 1
#epoch_size = 5
#state_size = 4

# epoch_size=1, 不能学习到任何一条依赖. 交叉熵接近0.66
#num_epoches = 2
#epoch_size = 1
#state_size = 1

# epoch_size=10, 可以学习到两条依赖. 交叉熵接近0.45
num_epoches = 10
epoch_size = 10
state_size = 16
