import utils
import models

import argparse
import json
import numpy as np
import os
import torch
from tqdm import tqdm
import pickle

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="cifar.json")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        parser.set_defaults(**json.load(f))
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    np.random.seed(args.seed)

    training_images = sorted(os.listdir(args.dataset_directory))
    images_path = [os.path.join(args.dataset_directory, img_name) for img_name in tqdm(training_images, desc="Selecting training images")]

    training_images_path = np.random.choice(images_path, args.train_size, replace=False)

    testing_images_path = np.setdiff1d(images_path, training_images_path)
    testing_images_path = np.random.choice(testing_images_path, args.test_size, replace=False)

    X, Y = utils.load_dataset(training_images_path)
    X, Y = X.to(args.device), Y.to(args.device)

    X_testing, Y_testing = utils.load_dataset(testing_images_path)
    X_testing, Y_testing = X_testing.to(args.device), Y_testing.to(args.device)

    compression_model = models.Trainer(
        size=args.test_size,
        dim_in=args.dim_in,
        dim_fourier=args.dim_fourier,
        dim_hidden=args.dim_hidden,
        dim_out=args.dim_out,
        num_layers=args.num_layers,
        std_init=args.log_std_init,
    ).to(args.device)


    state_dict_map = {
        "priors_mu.0": "layers.0.mu",
        "priors_mu.1": "layers.1.mu",
        "priors_mu.2": "layers.2.mu",
        "priors_mu.3": "layers.3.mu",
        "priors_std.0": "layers.0.log_std",
        "priors_std.1": "layers.1.log_std",
        "priors_std.2": "layers.2.log_std",
        "priors_std.3": "layers.3.log_std",
    }
    prior_state_dict = torch.load(os.path.join("Cifar_num4_emd32_lat16_beta2e-05", "model_prior_latest.pt"))
    prior_state_dict = dict((state_dict_map[key], value) for key, value in prior_state_dict.items())
    compression_model.prior.load_state_dict(prior_state_dict)
    #compression_model.train(X_testing, Y_testing, args.comp_epochs, args.lr, args.kl_beta)
    #print(compression_model.calculate_pnsr(X_testing, Y_testing))
    #print(compression_model.calculate_bpp(X_testing, Y_testing))
    #print()

    encoder = models.Encoder(args, compression_model, kl_beta=args.kl_beta)

    with open(os.path.join("Cifar_num4_emd32_lat16_beta2e-05", "groups.pkl"), "rb") as f:
        encoder.groups = pickle.load(f)
        print(len(encoder.groups))
        exit()


    encoder.train(X_testing, Y_testing, args.comp_epochs, args.lr)
    print(encoder.calculate_pnsr(X_testing, Y_testing))
    print(len(encoder.groups) * args.kl2_budget / 32 / 32)
    print()

    encoder.progressive_encode(X_testing, Y_testing, args.lr)

    print(encoder.calculate_pnsr(X_testing, Y_testing))
    print(len(encoder.groups) * args.kl2_budget / 32 / 32)
    print()


if __name__ == "__main__":
    main()
