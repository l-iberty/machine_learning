import os
import torch
import torchvision
from torch.utils.data import DataLoader
from model import VAE
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)

train_data = torchvision.datasets.MNIST(
    "mnist", train=True, transform=torchvision.transforms.ToTensor(), download=False)

test_data = torchvision.datasets.MNIST(
    "mnist", train=False, transform=torchvision.transforms.ToTensor(), download=False)

print("size of train_data: ", train_data.train_data.size())
print("size of test_data: ", test_data.test_data.size())

net_arch = {
    "n_hidden_encoder1": 500,
    "n_hidden_encoder2": 500,
    "n_hidden_decoder1": 500,
    "n_hidden_decoder2": 500,
    "n_z": 20,
    "n_input": 784,  # 28 * 28
}
lr = 0.001
batch_size = 100
device = "cpu"
model_save_path = "model_cpu.pt"
model_2d_save_path = "model_2d_cpu.pt"
gen_images_path="gen_images.pt"

vae_model = VAE(net_arch, lr, batch_size, device)
print(vae_model)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

if not os.path.exists(model_save_path):
    vae_model.train()
    n_epoch = 50
    for epoch in range(n_epoch):
        total_loss = 0.0
        count = 0
        for imgs, labels in train_loader:
            imgs = imgs.squeeze()  # [100,28,28]
            imgs = imgs.reshape(imgs.shape[0], -1)  # [100, 784]
            loss = vae_model.train_step(imgs.to(device))
            count += 1
            total_loss += loss
        print(f"epoch {epoch + 1} loss {total_loss / count}")

    torch.save(vae_model.state_dict(), model_save_path)
    print(f"vae_model saved in {model_save_path}")
else:
    print(f"load vae_model {model_save_path}")
    vae_model.load_state_dict(torch.load(model_save_path))

x, test_labels = next(iter(test_loader))
x = x.squeeze()
x = x.reshape(x.shape[0], -1)
x_reconstr = vae_model.reconstruct(x.to(device))

x = x.numpy()
x_reconstr = x_reconstr.cpu().detach().numpy()

""" check reconstruction
"""
plt.figure(figsize=(6, 10))
rows, cols = 5, 2
for i in range(rows):
    plt.subplot(rows, cols, 2 * i + 1)
    plt.imshow(x[i].reshape(28, 28), vmin=0, vmax=1, cmap="Greys_r")
    plt.title("test input")
    plt.colorbar()
    plt.subplot(rows, cols, 2 * i + 2)
    plt.imshow(x_reconstr[i].reshape(28, 28), vmin=0, vmax=1, cmap="Greys_r")
    plt.title("reconstruct")
    plt.colorbar()
    plt.tight_layout()
plt.show()

""" check generation
"""
vae_model.eval()
noise = torch.randn(batch_size, net_arch["n_z"], device=device)
images = vae_model.generate(noise).cpu().detach()
torch.save(images, gen_images_path)
print(f"{gen_images_path} saved")

""" check latent space, in order to do that, we need to train another VAE model with n_z=2
"""
net_arch["n_z"] = 2

vae_model_2d = VAE(net_arch, lr, batch_size, device)
print(vae_model_2d)

if not os.path.exists(model_2d_save_path):
    vae_model_2d.train()
    n_epoch = 50
    for epoch in range(n_epoch):
        total_loss = 0.0
        count = 0
        for imgs, labels in train_loader:
            imgs = imgs.squeeze()  # [100,28,28]
            imgs = imgs.reshape(imgs.shape[0], -1)  # [100, 784]
            loss = vae_model_2d.train_step(imgs.to(device))
            count += 1
            total_loss += loss
        print(f"epoch {epoch + 1} loss {total_loss / count}")

    torch.save(vae_model_2d.state_dict(), model_2d_save_path)
    print(f"vae_model saved in {model_2d_save_path}")
else:
    print(f"load vae_model {model_2d_save_path}")
    vae_model_2d.load_state_dict(torch.load(model_2d_save_path))

nx = ny = 20
x_values = np.linspace(-3, 3, nx)
y_values = np.linspace(-3, 3, nx)
plt.figure(figsize=(8, 10))
canvas = np.zeros(shape=(28 * nx, 28 * ny))
for i, xi in enumerate(x_values):
    for j, yj in enumerate(y_values):
        z_mu = np.array([[xi, yj]] * batch_size, dtype=np.float32)
        x_mean = vae_model_2d.generate(torch.from_numpy(z_mu).to(device)).cpu().detach().numpy()
        canvas[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = x_mean[0].reshape(28, 28)

plt.imshow(canvas, origin="upper", cmap="Greys_r")
plt.tight_layout()
plt.show()

test_loader = DataLoader(test_data, batch_size=5000, shuffle=True)
x, test_labels = next(iter(test_loader))
x = x.squeeze()
x = x.reshape(x.shape[0], -1)
z_mu = vae_model_2d.transform(x.to(device))
z_mu = z_mu.cpu().detach().numpy()
plt.scatter(x=z_mu[:, 0], y=z_mu[:, 1], c=test_labels.numpy())
plt.colorbar()
plt.grid()
plt.show()


