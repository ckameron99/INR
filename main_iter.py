import argparse
import json
import random

import numpy as np
import torch

import ml_kit


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

    X, Y, X_testing, Y_testing = ml_kit.load_train_test_dataset(args)

    while True:
        args.kl_beta = 10 ** random.uniform(-7, -4)
        args.log_std_init = -5.8
        trained_prior = ml_kit.train_prior(args, X, Y)
        trained_prior.gen_groups(args.kl2_budget, args.max_group_size)
        args.log_std_init = -5.8
        encoder = ml_kit.train_encoder(args, X_testing, Y_testing, trained_prior=trained_prior)

        print(args.kl_beta)
        print(encoder.trainer.calculate_pnsr(X_testing, Y_testing).mean())
        print(len(encoder.trainer.groups) * args.kl2_budget / 32 / 32)
        print()
        torch.cuda.empty_cache()




if __name__ == "__main__":
    main()
