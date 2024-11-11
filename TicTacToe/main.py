import argparse

from train import train


def main():
    parser = argparse.ArgumentParser(description="Train or Evaluate a model")
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--eval', action='store_true',
                        help='Evaluate the model')
    parser.add_argument('--model_dir', type=str, default="N/A",
                        help='Directory where the model is saved '
                        'or loaded from')
    parser.print_help()

    args = parser.parse_args()

    # Check that either --train or --eval is provided, but not both or neither
    if args.train == args.eval:
        parser.error('You must specify either --train or --eval,'
                     'but not both or none.')

    if args.train:
        print("Training the model...")
        # Insert training code here
        train(args)

    if args.eval:
        print("Evaluating the model...")
        # Insert evaluation code here


if __name__ == "__main__":
    main()
