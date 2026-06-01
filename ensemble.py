"""Blend experiment probabilities and create a validated Kaggle submission."""

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
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_probabilities(output_dir: Path, split: str) -> np.ndarray:
    path = output_dir / f"{split}_probabilities.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Missing probability file: {path}")
    return np.load(path)


def calculate_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    one_hot = np.eye(len(CLASSES))[labels]
    return float(roc_auc_score(one_hot, probabilities, average="macro", multi_class="ovr"))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    val_df = pd.read_csv(find_csv(args.data_dir, "val_data.csv"))
    test_df = pd.read_csv(find_csv(args.data_dir, "test_data.csv"))
    sample_path = find_csv(args.data_dir, "sample_submission.csv")
    labels = val_df["AQI_Class"].map(CLASS_TO_IDX).to_numpy()

    first_val = load_probabilities(args.first_output_dir, "val")
    second_val = load_probabilities(args.second_output_dir, "val")
    first_test = load_probabilities(args.first_output_dir, "test")
    second_test = load_probabilities(args.second_output_dir, "test")

    results = []
    for first_weight in np.linspace(0.0, 1.0, 21):
        probabilities = first_weight * first_val + (1.0 - first_weight) * second_val
        results.append(
            {
                "first_weight": float(first_weight),
                "second_weight": float(1.0 - first_weight),
                "val_macro_roc_auc": calculate_auc(labels, probabilities),
            }
        )
    best = max(results, key=lambda row: row["val_macro_roc_auc"])
    test_probabilities = (
        best["first_weight"] * first_test + best["second_weight"] * second_test
    )
    write_submission(test_probabilities, test_df, sample_path, args.output_dir)
    (args.output_dir / "ensemble_metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(best, indent=2))
    print(f"Wrote {args.output_dir / 'submission.csv'}")


if __name__ == "__main__":
    main()

