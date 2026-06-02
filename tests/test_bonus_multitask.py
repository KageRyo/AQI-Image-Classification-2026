from __future__ import annotations

import numpy as np
import pandas as pd

from bonus_multitask import TARGETS, TargetNormalizer, write_bonus_csvs


def make_target_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "AQI": [50.0, 150.0, 250.0],
            "PM2.5": [10.0, 50.0, 100.0],
            "PM10": [20.0, 80.0, 160.0],
            "O3": [5.0, np.nan, 45.0],
            "CO": [1.0, 20.0, 100.0],
            "SO2": [2.0, 8.0, 20.0],
            "NO2": [3.0, 15.0, 40.0],
        }
    )


def test_mask_strategy_excludes_missing_targets_from_loss() -> None:
    dataframe = make_target_dataframe()
    normalizer = TargetNormalizer.from_dataframe(dataframe, "mask")

    transformed, mask = normalizer.transform(dataframe[TARGETS].to_numpy(dtype=np.float32))

    assert np.isfinite(transformed).all()
    assert mask[1, TARGETS.index("O3")] == 0


def test_imputation_strategies_include_missing_targets_in_loss() -> None:
    dataframe = make_target_dataframe()
    values = dataframe[TARGETS].to_numpy(dtype=np.float32)

    for strategy in ["mean", "median"]:
        normalizer = TargetNormalizer.from_dataframe(dataframe, strategy)
        transformed, mask = normalizer.transform(values)
        restored = normalizer.inverse_transform(transformed)

        assert np.isfinite(restored).all()
        assert mask[1, TARGETS.index("O3")] == 1


def test_write_bonus_csvs_uses_required_columns(tmp_path) -> None:
    test = pd.DataFrame({"Filename": ["first.jpg", "second.jpg"]})
    predictions = np.arange(2 * len(TARGETS), dtype=np.float32).reshape(2, len(TARGETS))

    write_bonus_csvs(test, predictions, tmp_path)

    minimal = pd.read_csv(tmp_path / "bonus_aqi_pm25.csv")
    complete = pd.read_csv(tmp_path / "bonus_all_metrics.csv")
    assert minimal.columns.tolist() == ["Filename", "AQI", "PM2.5"]
    assert complete.columns.tolist() == ["Filename", *TARGETS]
    assert complete["Filename"].tolist() == test["Filename"].tolist()
