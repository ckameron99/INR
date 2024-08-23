import argparse
import json

import numpy as np

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

    trained_prior = ml_kit.train_prior(args, X, Y)
    trained_prior.gen_groups(args.kl2_budget, args.max_group_size)

    encoder = ml_kit.train_encoder(args, X_testing, Y_testing, trained_prior=trained_prior)

    print(encoder.trainer.calculate_pnsr(X_testing, Y_testing).mean())
    print(len(encoder.trainer.groups) * args.kl2_budget / 32 / 32)
    print()




if __name__ == "__main__":
    main()
