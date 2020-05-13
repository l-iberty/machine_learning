import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.xavier_uniform_(m.weight.data, gain=0.5)  # xavier 初始化是非常有必要的, 它可以防止 input 随着传递层数的增加而变得越来越大
        nn.init.constant_(m.bias.data, 0)


class VAE(nn.Module):
    def __init__(self, net_arch, lr=0.001, batch_size=100, device="cpu"):
        super(VAE, self).__init__()

        self.net_arch = net_arch
        self.lr = lr
        self.batch_size = batch_size
        self.device = device

        self.encoder = Encoder(net_arch, device)
        self.decoder = Decoder(net_arch, device)

        parameters = list(self.parameters()) + list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=lr, betas=(0.9, 0.999))

        # self.reconstr_criterion = nn.MSELoss(size_average=False)
        self.reconstr_criterion = nn.BCELoss(size_average=False)

    def forward(self):
        pass

    def train_step(self, x):
        z_mu, z_logvar = self.encoder(x)
        z = self.reparameterize(z_mu, z_logvar, self.device)
        x_reconstr_mean = self.decoder(z)

        reconstr_loss = self.reconstr_criterion(x_reconstr_mean, x)
        KLD = -0.5 * torch.sum(1 + z_logvar - z_logvar.exp() - z_mu.pow(2))

        self.optimizer.zero_grad()

        loss = reconstr_loss + KLD
        loss.backward()

        self.optimizer.step()
        return loss.item()

    def reparameterize(self, mu, logvar, device):
        epsilon = torch.FloatTensor(self.batch_size, self.net_arch["n_z"]).to(device)
        nn.init.normal_(epsilon, mean=0, std=1)
        sigma = torch.sqrt(torch.exp(logvar))
        return mu + epsilon * sigma

    def reconstruct(self, x):
        z_mu, z_logvar = self.encoder(x)
        z = self.reparameterize(z_mu, z_logvar, self.device)
        return self.decoder(z)

    def generate(self, z_mu):
        return self.decoder(z_mu)

    def transform(self, x):
        z_mu, _ = self.encoder(x)
        return z_mu


class Encoder(nn.Module):
    def __init__(self, net_arch, device="cpu"):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(net_arch["n_input"], net_arch["n_hidden_encoder1"]),
            nn.ReLU(),
            nn.Linear(net_arch["n_hidden_encoder1"], net_arch["n_hidden_encoder2"]),
            nn.ReLU(),
        )
        self.z_mu = nn.Linear(net_arch["n_hidden_encoder2"], net_arch["n_z"])
        self.z_logvar = nn.Linear(net_arch["n_hidden_encoder2"], net_arch["n_z"])
        self.apply(weights_init)
        self.to(device)

    def forward(self, x):  # x: [batch_size, n_input]
        h = self.encoder(x)
        z_mu = self.z_mu(h)
        z_logvar = self.z_mu(h)
        return z_mu, z_logvar


class Decoder(nn.Module):
    def __init__(self, net_arch, device="cpu"):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(net_arch["n_z"], net_arch["n_hidden_decoder1"]),
            nn.ReLU(),
            nn.Linear(net_arch["n_hidden_decoder1"], net_arch["n_hidden_decoder2"]),
            nn.ReLU(),
            nn.Linear(net_arch["n_hidden_decoder2"], net_arch["n_input"]),
            nn.Sigmoid(),
        )
        self.apply(weights_init)
        self.to(device)

    def forward(self, z):  # z: [batch_size, n_z]
        return self.decoder(z)

