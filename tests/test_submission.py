from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from main import CLASSES, validate_csvs, write_submission


def make_labeled_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Filename": ["first.jpg", "second.jpg"],
            "AQI_Class": [CLASSES[0], CLASSES[-1]],
        }
    )


def test_validate_csvs_rejects_test_labels() -> None:
    labeled = make_labeled_dataframe()
    test = pd.DataFrame({"Filename": ["test.jpg"], "AQI_Class": [CLASSES[0]]})

    with pytest.raises(ValueError, match="hidden test labels"):
        validate_csvs(labeled, labeled, test)


def test_write_submission_matches_sample_columns(tmp_path) -> None:
    sample_path = tmp_path / "sample_submission.csv"
    pd.DataFrame(columns=["Filename", *CLASSES]).to_csv(sample_path, index=False)
    test = pd.DataFrame({"Filename": ["first.jpg", "second.jpg"]})
    probabilities = np.array(
        [
            [0.1, 0.2, 0.3, 0.15, 0.15, 0.1],
            [0.05, 0.05, 0.1, 0.2, 0.3, 0.3],
        ]
    )

    write_submission(probabilities, test, sample_path, tmp_path)

    submission = pd.read_csv(tmp_path / "submission.csv")
    assert submission.columns.tolist() == ["Filename", *CLASSES]
    assert submission["Filename"].tolist() == test["Filename"].tolist()
    assert np.allclose(submission[CLASSES].sum(axis=1), 1.0)


def test_write_submission_rejects_invalid_probability_sums(tmp_path) -> None:
    sample_path = tmp_path / "sample_submission.csv"
    pd.DataFrame(columns=["Filename", *CLASSES]).to_csv(sample_path, index=False)
    test = pd.DataFrame({"Filename": ["test.jpg"]})

    with pytest.raises(ValueError, match="sum to one"):
        write_submission(np.full((1, len(CLASSES)), 0.1), test, sample_path, tmp_path)
