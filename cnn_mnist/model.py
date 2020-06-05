import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, recall_score, f1_score


class CNN(nn.Module):
    def __init__(self, filter_shape, kernel_size, nclasses, device):
        super(CNN, self).__init__()

        padding = (kernel_size - 1) // 2
        self.conv = nn.Sequential(  # input shape [1,28,28]
            nn.Conv2d(in_channels=1, out_channels=filter_shape[0],
                      kernel_size=kernel_size, stride=1,
                      padding=padding),  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            # output shape [filter_shape[0],28,28]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # output shape [filter_shape[0], 28/2, 28/2] => [filter_shape[0],14,14]

            nn.Conv2d(in_channels=filter_shape[0], out_channels=filter_shape[1],
                      kernel_size=kernel_size, stride=1,
                      padding=padding),
            # output shape [filter_shape[1],14,14]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # output shape [filter_shape[1],7,7]
        )
        self.linear = nn.Linear(filter_shape[1] * 7 * 7, nclasses)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss(size_average=False)

        self.device = device
        self.to(device)

    def forward(self, x):  # x [batch_size, 1, 28, 28]
        x = x.to(self.device)
        x = self.conv(x)  # [batch_size, filter_shape[1], 7, 7]
        x = x.reshape(x.size(0), -1)  # [batch_size, filter_shape[1] × 7 × 7]
        x = self.linear(x)  # [batch_size, nclasses]
        return x

    def train_step(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)

        output = self.forward(x)
        self.optimizer.zero_grad()
        loss = self.criterion(output, y)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)

        output = self.forward(x)
        pred_y = torch.max(output, 1)[1]

        y = y.cpu().detach().numpy()
        pred_y = pred_y.cpu().detach().numpy()
        # print('real', y)
        # print('pred_y', pred_y)
        print('P', accuracy_score(y, pred_y))
        print('R', recall_score(y, pred_y, average='macro'))
        print('F1', f1_score(y, pred_y, average='macro'))
