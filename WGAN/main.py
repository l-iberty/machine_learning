import os
import torch
import torchvision
from torch.utils.data import DataLoader
from model import WGAN
import numpy as np

np.set_printoptions(threshold=np.inf)

train_data = torchvision.datasets.MNIST(
    "mnist", train=True, transform=torchvision.transforms.ToTensor(), download=False)

test_data = torchvision.datasets.MNIST(
    "mnist", train=False, transform=torchvision.transforms.ToTensor(), download=False)

print("size of train_data: ", train_data.train_data.size())
print("size of test_data: ", test_data.test_data.size())

net_arch = {
    "n_hidden_1": 500,
    "n_hidden_2": 500,
    "n_z": 20,
    "n_input": 784,  # 28 * 28
}
lr = 0.001
batch_size = 100
device = "cpu"
model_save_path = "model_cpu.pt"
gen_images_path = "gen_images.pt"
n_epoch = 50
n_epoch_D = 5

wgan_model = WGAN(net_arch, lr, batch_size, device)
print(wgan_model)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

if not os.path.exists(model_save_path):
    wgan_model.train()
    for epoch in range(n_epoch):
        D_losses, G_losses = [], []

        for imgs, labels in train_loader:
            imgs = imgs.squeeze()  # [100,28,28]
            imgs = imgs.reshape(imgs.shape[0], -1)  # [100, 784]

            D_losses_tmp = []
            for _ in range(n_epoch_D):
                noise = torch.randn(batch_size, net_arch["n_z"], device=device)  # [100, 20]
                D_loss = wgan_model.train_step_D(imgs.to(device), noise)
                D_losses_tmp.append(D_loss)

            D_losses.append(np.mean(D_losses_tmp))

            noise = torch.randn(batch_size, net_arch["n_z"], device=device)  # [100, 20]
            G_loss = wgan_model.train_step_G(noise)
            G_losses.append(G_loss)

        print(f"Epoch {epoch + 1} D_loss {D_losses[-1]}, G_loss {G_losses[-1]}")

    torch.save(wgan_model.state_dict(), model_save_path)
    print(f"wgan_model saved in {model_save_path}")
else:
    wgan_model.load_state_dict(torch.load(model_save_path))
    print(f"wgan_model loaded from {model_save_path}")

wgan_model.eval()
noise = torch.randn(batch_size, net_arch["n_z"], device=device)
images = wgan_model.generate(noise).cpu().detach()
torch.save(images, gen_images_path)
print(f"{gen_images_path} saved")
