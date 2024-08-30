import argparse
from dep.SetCreator import *


parser = argparse.ArgumentParser()

parser.add_argument("--data_path", type=str, required=True, help="Data location")


def main(args):
    # Create instance of training/validation loader
    train_loader, valid_loader = create_training_validation_set(args.data_path, download=True)

    # Create instance of test loader
    test_loader = create_testing_set(args.data_path, download=True)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
    print("done!")
