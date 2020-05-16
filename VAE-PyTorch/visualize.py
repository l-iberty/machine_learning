import torch
import numpy as np
import matplotlib.pyplot as plt

gen_images_path = "gen_images.pt"

images = torch.load(gen_images_path)  # [batch_size, n_input]

nx = ny = 10
plt.figure(figsize=(6, 6))
canvas = np.zeros(shape=(28 * nx, 28 * ny))
i = 0
for x in range(nx):
    for y in range(ny):
        canvas[x * 28:(x + 1) * 28, y * 28:(y + 1) * 28] = images[i].reshape(28, 28)
        i += 1

plt.xticks([])
plt.yticks([])
plt.imshow(canvas, cmap="Greys_r")
plt.tight_layout()
plt.show()
