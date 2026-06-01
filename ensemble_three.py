"""Blend three experiment probability files and create a validated submission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from main import CLASSES, CLASS_TO_IDX, find_csv, write_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--first-output-dir", type=Path, required=True)
    parser.add_argument("--second-output-dir", type=Path, required=True)
    parser.add_argument("--third-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, default=5, help="Weight step as an integer percentage.")
    return parser.parse_args()


def load_probabilities(output_dir: Path, split: str) -> np.ndarray:
    path = output_dir / f"{split}_probabilities.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Missing probability file: {path}")
    return np.load(path)


def main() -> None:
    args = parse_args()
    if not 1 <= args.step <= 100:
        raise ValueError("--step must be between 1 and 100.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    val_df = pd.read_csv(find_csv(args.data_dir, "val_data.csv"))
    test_df = pd.read_csv(find_csv(args.data_dir, "test_data.csv"))
    sample_path = find_csv(args.data_dir, "sample_submission.csv")
    labels = val_df["AQI_Class"].map(CLASS_TO_IDX).to_numpy()
    one_hot = np.eye(len(CLASSES))[labels]
    output_dirs = [
        args.first_output_dir,
        args.second_output_dir,
        args.third_output_dir,
    ]
    val_probabilities = [load_probabilities(path, "val") for path in output_dirs]
    test_probabilities = [load_probabilities(path, "test") for path in output_dirs]

    results = []
    for first_percent in range(0, 101, args.step):
        for second_percent in range(0, 101 - first_percent, args.step):
            third_percent = 100 - first_percent - second_percent
            weights = np.array([first_percent, second_percent, third_percent]) / 100
            blended = sum(weight * probabilities for weight, probabilities in zip(weights, val_probabilities))
            results.append(
                {
                    "weights": weights.tolist(),
                    "val_macro_roc_auc": float(
                        roc_auc_score(one_hot, blended, average="macro", multi_class="ovr")
                    ),
                }
            )
    results.sort(key=lambda row: row["val_macro_roc_auc"], reverse=True)
    best = results[0]
    blended_test = sum(
        weight * probabilities for weight, probabilities in zip(best["weights"], test_probabilities)
    )
    write_submission(blended_test, test_df, sample_path, args.output_dir)
    (args.output_dir / "ensemble_metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(best, indent=2))
    print(f"Wrote {args.output_dir / 'submission.csv'}")


if __name__ == "__main__":
    main()

