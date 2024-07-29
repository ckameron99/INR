from PIL import Image
from torchvision.transforms import ToTensor
import torch


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