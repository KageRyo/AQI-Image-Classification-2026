"""Blend test probability files with explicit weights and validate the submission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from main import find_csv, write_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--weights", type=float, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_test_probabilities(output_dir: Path) -> np.ndarray:
    path = output_dir / "test_probabilities.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Missing probability file: {path}")
    probabilities = np.load(path)
    if probabilities.ndim != 2:
        raise ValueError(f"Expected a two-dimensional probability array: {path}")
    return probabilities


def main() -> None:
    args = parse_args()
    if len(args.output_dirs) != len(args.weights):
        raise ValueError("--output-dirs and --weights must have the same length.")
    weights = np.asarray(args.weights, dtype=np.float64)
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("--weights must be non-negative and sum to a positive value.")
    weights /= weights.sum()
    probabilities = [load_test_probabilities(path) for path in args.output_dirs]
    if len({array.shape for array in probabilities}) != 1:
        raise ValueError("All test probability arrays must have the same shape.")

    blended = sum(weight * array for weight, array in zip(weights, probabilities))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    test_df = pd.read_csv(find_csv(args.data_dir, "test_data.csv"))
    sample_path = find_csv(args.data_dir, "sample_submission.csv")
    write_submission(blended, test_df, sample_path, args.output_dir)
    metadata = {
        "output_dirs": [str(path) for path in args.output_dirs],
        "weights": weights.tolist(),
    }
    (args.output_dir / "blend_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))
    print(f"Wrote {args.output_dir / 'submission.csv'}")


if __name__ == "__main__":
    main()
