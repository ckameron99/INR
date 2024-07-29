import argparse
import json

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
    print(args.dataset_location)


if __name__ == "__main__":
    main()