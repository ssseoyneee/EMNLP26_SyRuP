import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from syrup.config import get_args


def main():
    args = get_args()
    if args.stage == "train":
        from syrup.train import train
        train(args)
    else:
        from syrup.inference import inference
        inference(args)


if __name__ == "__main__":
    main()
