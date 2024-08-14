import utils
import models

import argparse
import json
import numpy as np
import os
from tqdm import tqdm

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

    prior_model = models.Trainer(
        size=args.train_size,
        dim_in=args.dim_in,
        dim_fourier=args.dim_fourier,
        dim_hidden=args.dim_hidden,
        dim_out=args.dim_out,
        num_layers=args.num_layers,
        std_init=args.log_std_init,
    ).to(args.device)

    """print(prior_model.calculate_pnsr(X, Y))
    print(prior_model.calculate_bpp(X, Y))
    prior_model.update_prior()
    print(prior_model.calculate_bpp(X, Y))
    utils.cum_dist(prior_model)
    exit()"""

    lr = args.lr
    kl_beta = args.kl_beta
    prior_model.train(X, Y, args.n_epochs*2, lr, kl_beta)
    for _ in tqdm(range(args.n_em_iters)):
        prior_model.train(X, Y, args.n_epochs, lr, kl_beta)
        prior_model.update_prior()

    print(prior_model.calculate_pnsr(X, Y))
    print(prior_model.calculate_bpp(X, Y))

    compression_model = models.Trainer(
        size=args.test_size,
        dim_in=args.dim_in,
        dim_fourier=args.dim_fourier,
        dim_hidden=args.dim_hidden,
        dim_out=args.dim_out,
        num_layers=args.num_layers,
        std_init=args.log_std_init,
    ).to(args.device)


    compression_model.prior.load_state_dict(prior_model.prior.state_dict())
    compression_model.train(X_testing, Y_testing, 30, lr, kl_beta)
    print(compression_model.calculate_pnsr(X_testing, Y_testing))
    print(compression_model.calculate_bpp(X_testing, Y_testing))

    encoder = models.Encoder(compression_model)




if __name__ == "__main__":
    main()
