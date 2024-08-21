import utils
import models
from tqdm import tqdm
import numpy as np
import os
import torch
from PIL import Image
from torchvision.transforms import ToTensor

def train_encoder(args, X_testing, Y_testing, compression_model=None, trained_prior=None):
    if compression_model is None and trained_prior is None:
        raise ValueError("Either compression model or trained_prior must not be none")
    elif compression_model is not None and trained_prior is not None:
        raise ValueError("Compression model and trained_prior may be incompatible, please only pass one")
    elif compression_model is None:
        compression_model = train_compression_model(args, X_testing, Y_testing, trained_prior)
    encoder = models.Encoder(args, compression_model, kl_beta=args.kl_beta)
    encoder.train(X_testing, Y_testing, args.comp_epochs, args.lr)

    encoder.progressive_encode(X_testing, Y_testing, args.lr)

    return encoder

def train_compression_model(args, X_testing, Y_testing, prior_model):
    compression_model = models.Trainer(
        args,
        size=args.test_size
    ).to(args.device)


    compression_model.prior.load_state_dict(prior_model.prior.state_dict())
    compression_model.train(X_testing, Y_testing, args.comp_epochs, args.lr, args.kl_beta)

    return compression_model

def train_prior(args, X, Y):
    prior_model = models.Trainer(
        args,
        size=args.train_size,
    ).to(args.device)

    prior_model.train(X, Y, args.n_epochs*2, args.lr, args.kl_beta)
    for _ in tqdm(range(args.n_em_iters)):
        prior_model.train(X, Y, args.n_epochs, args.lr, args.kl_beta)
        prior_model.update_prior()

    return prior_model

def load_train_test_dataset(args):
    training_images = sorted(os.listdir(args.dataset_directory))
    images_path = [os.path.join(args.dataset_directory, img_name) for img_name in tqdm(training_images, desc="Selecting training images")]

    training_images_path = np.random.choice(images_path, args.train_size, replace=False)

    testing_images_path = np.setdiff1d(images_path, training_images_path)
    testing_images_path = np.random.choice(testing_images_path, args.test_size, replace=False)

    X, Y = load_dataset(training_images_path)
    X, Y = X.to(args.device), Y.to(args.device)

    X_testing, Y_testing = load_dataset(testing_images_path)
    X_testing, Y_testing = X_testing.to(args.device), Y_testing.to(args.device)

    return X, Y, X_testing, Y_testing

def load_dataset(images_path):
    if not len(images_path):
        return torch.tensor([]), torch.tensor([])
    X = []
    Y = []
    for path in images_path:
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
