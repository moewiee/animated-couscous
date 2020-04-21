import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="",
                        help="config yaml path")
    parser.add_argument("--load", type=str, default="",
                        help="path to model weight")
    parser.add_argument("--mode", type=str, default="train",
                        help="model running mode (train/valid/test)")
    # parser.add_argument("--fold", type=int, default=0,
    #                     help="validation data fold")
    parser.add_argument("--valid", action="store_true",
                        help="enable evaluation mode for validation")
    parser.add_argument("--test", action="store_true",
                        help="enable evaluation mode for testset")
    parser.add_argument("--holdout", action="store_true",
                        help="enable evaluation mode for holdout set")
    parser.add_argument("--reset", action="store_true",
                        help="reset epoch and best metric")
    parser.add_argument("--swa", action="store_true",
                        help="finetune swa cycle")
    parser.add_argument("--debug", action="store_true",
                        help="enable debug mode for test")

    args = parser.parse_args()
    if args.valid:
        args.mode = "valid"
    elif args.test:
        args.mode = "test"
    elif args.swa:
        args.mode = "swa"
    elif args.holdout:
        args.mode = "holdout"

    return args
