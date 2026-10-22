import argparse

from experiment.engine import load_config, train_model
from experiment.models import MODEL_NAMES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_NAMES, default="fact_ednet")
    parser.add_argument("--config", default="configs/experiment1.json")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    train_model(args.model, args.manifest, args.data_root, load_config(args.config), args.output, args.seed, args.device)


if __name__ == "__main__":
    main()
