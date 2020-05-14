import torch
import torch.nn as nn
import torch.optim as optim


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.xavier_uniform_(m.weight.data, gain=0.5)  # xavier 初始化是非常有必要的, 它可以防止 input 随着传递层数的增加而变得越来越大
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, net_arch, device="cpu"):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(net_arch["n_z"], net_arch["n_hidden_1"]),
            nn.ReLU(),
            nn.Linear(net_arch["n_hidden_1"], net_arch["n_hidden_2"]),
            nn.ReLU(),
            nn.Linear(net_arch["n_hidden_2"], net_arch["n_input"]),
            nn.Sigmoid(),
        )
        self.apply(weights_init)
        self.to(device)

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, net_arch, device="cpu"):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(net_arch["n_input"], net_arch["n_hidden_1"]),
            nn.ReLU(),
            nn.Linear(net_arch["n_hidden_1"], net_arch["n_hidden_2"]),
            nn.ReLU(),
            nn.Linear(net_arch["n_hidden_2"], net_arch["n_z"]),
            nn.Sigmoid(),
        )
        self.apply(weights_init)
        self.to(device)

    def forward(self, x):
        return self.main(x)


class WGAN(nn.Module):
    def __init__(self, net_arch, lr=0.001, batch_size=100, device="cpu"):
        super(WGAN, self).__init__()

        self.net_arch = net_arch
        self.lr = lr
        self.batch_size = batch_size
        self.device = device

        self.G = Generator(net_arch, device)
        self.D = Discriminator(net_arch, device)

        self.optimizerG = optim.Adam(self.G.parameters(), lr=lr)
        self.optimizerD = optim.Adam(self.D.parameters(), lr=lr)

    def forward(self):
        pass

    def train_step_D(self, x, noise):
        """ x: [batch_size, n_input]
        """
        # D(x) 是 D 判断出来的 x 是真实样本的概率, 所以 D(x) 要尽可能接近1
        # D(G(z)) 是 D 判断出来的 G(z) 是真实样本的概率(z就是noise), 所以 D(G(z)) 要尽可能接近0
        G_z = self.G(noise)  # [batch_size, n_input]
        D_real_score = self.D(x)  # [batch_size, n_z]
        D_fake_score = self.D(G_z)  # [batch_size, n_z]

        self.optimizerD.zero_grad()

        # WGAN loss for D:
        # 最小化 -E[G(x)] + E[D(G(z))]
        D_loss = -torch.mean(D_real_score) + torch.mean(D_fake_score)

        D_loss.backward()
        self.optimizerD.step()

        clip = 0.01
        for param in self.D.parameters():
            param.data.clamp_(-clip, clip)

        return D_loss.item()

    def train_step_G(self, noise):
        """noise: [batch_size, n_z]
        """
        # D(G(z)) 越接近1就说明 G 生成的 fake data 骗过了 D
        G_z = self.G(noise)
        D_fake_score = self.D(G_z)

        self.optimizerG.zero_grad()

        # WGAN loss for G
        # 最小化 -E[D(G(z))]
        G_loss = -torch.mean(D_fake_score)

        G_loss.backward()
        self.optimizerG.step()

        return G_loss.item()

    def generate(self, noise):
        return self.G(noise)
