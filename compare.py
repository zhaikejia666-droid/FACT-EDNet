import argparse
from pathlib import Path

import numpy as np

from experiment.engine import evaluate_model, load_config, train_model
from experiment.models import MODEL_NAMES
from experiment.utils import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment1.json")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    root = Path(args.output)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Comparison output must be empty")
    config = load_config(args.config)
    runs = []
    for name in dict.fromkeys(args.models):
        for seed in config["seeds"]:
            directory = root / name / f"seed_{seed}"
            checkpoint = train_model(name, args.manifest, args.data_root, config, directory, seed, args.device)
            runs.append((name, seed, checkpoint, directory))
    results = {name: [] for name in dict.fromkeys(args.models)}
    for name, seed, checkpoint, directory in runs:
        results[name].append(evaluate_model(checkpoint, args.manifest, args.data_root, directory / "test.json", args.device))
    summary = {}
    for name, rows in results.items():
        summary[name] = {"runs": rows}
        for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
            values = np.array([row[key] for row in rows])
            summary[name][key] = {"mean": float(values.mean()), "sample_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0}
    write_json(root / "comparison.json", summary)


if __name__ == "__main__":
    main()
