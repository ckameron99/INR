import hashlib
from PIL import Image
from torchvision.transforms import ToTensor
import torch
import matplotlib.pyplot as plt
import numpy as np


def load_dataset(training_images_path):
    if not len(training_images_path):
        return torch.tensor([]), torch.tensor([])
    X = []
    Y = []
    for path in training_images_path:
        image = Image.open(path)
        image = ToTensor()(image)
        y = image.reshape(image.shape[0], -1).T


        l_list = []
        for _, s in enumerate(image.shape[1:]):
            l = (1 + 2 * torch.arange(s)) / s
            l -= 1
            l_list.append(l)
        x = torch.meshgrid(*l_list, indexing="ij")
        x = torch.stack(x, dim=-1).view(-1, 2)

        X.append(x[None, :, :])
        Y.append(y[None, :, :])
    return torch.cat(X, dim=0), torch.cat(Y, dim=0)

def checksum(t):
    torch.manual_seed(69)
    t = torch.flatten(t)
    tb = t.cpu().detach().numpy().tobytes()
    sha256=hashlib.sha256()
    sha256.update(tb)
    return sha256.hexdigest()

def cum_dist(model):
    n = len(model.prior.layers)
    fig, axes = plt.subplots(n, 1, figsize=(5, 5*n))

    if n == 1:
        axes = [axes]

    for i in range(n):
        ax = axes[i]
        data = model.params[f"layers.{i}.mu"].clone().detach().cpu().flatten()
        count, bins_count = np.histogram(data, bins=100)
        pdf = count / sum(count)
        cdf = np.cumsum(pdf)
        ax.plot(bins_count[1:], cdf)
        ax.set_title(f"pdf for layer {i}")

    plt.tight_layout()
    plt.show()
