import os
import torch
import torchvision
from torch.utils.data import DataLoader
from model import CNN
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)

train_data = torchvision.datasets.MNIST(
    "mnist", train=True, transform=torchvision.transforms.ToTensor(), download=False)

test_data = torchvision.datasets.MNIST(
    "mnist", train=False, transform=torchvision.transforms.ToTensor(), download=False)

print("size of train_data: ", train_data.train_data.size())
print("size of test_data: ", test_data.test_data.size())

model_file = 'model.pt'
device = "cuda:0" if torch.cuda.is_available() else "cpu"
batch_size = 100

filter_shape = [16, 32]
kernel_size = 5
nclasses = 10

cnn_model = CNN(filter_shape, kernel_size, nclasses, device)
print(cnn_model)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

if not os.path.exists(model_file):
    cnn_model.train()
    losses = []
    for epoch in range(50):
        for imgs, labels in train_loader:  # imgs [batch_size, 1, 28, 28], labels [100]
            loss = cnn_model.train_step(imgs, labels)
            losses.append(loss)

        print('epoch {} loss {}'.format(epoch + 1, np.mean(losses)))

    torch.save(cnn_model.state_dict(), model_file)
else:
    cnn_model.load_state_dict(torch.load(model_file))

cnn_model.eval()
x, y = next(iter(test_loader))
cnn_model.evaluate(x, y)
