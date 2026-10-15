"""Leakage tests: features at a forecast origin must not depend on later sales."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from src.backtest import fold_origins, training_rows
from src.config import ModelConfig
from src.features import FEATURES, build_features, grid


def _features_at(arr, origin: int, horizon: int = 28) -> pd.DataFrame:
    it, og, hs = grid(len(arr.items), [origin], horizon)
    return build_features(arr, it, og, hs)[FEATURES]


def test_features_ignore_future_sales(synthetic_arrays) -> None:
    """Scrambling every sale after the origin must leave all features unchanged."""
    rng = np.random.default_rng(1)
    for origin in (30, 100, 250, 371):
        before = _features_at(synthetic_arrays, origin)
        corrupted = synthetic_arrays.sales.copy()
        # Column `origin` is day origin + 1, the first unknown day.
        corrupted[:, origin:] = rng.integers(0, 10_000, size=corrupted[:, origin:].shape)
        after = _features_at(replace(synthetic_arrays, sales=corrupted), origin)
        pd.testing.assert_frame_equal(before, after)


def test_features_do_see_the_origin_day(synthetic_arrays) -> None:
    """Control for the test above: changing the origin day itself must change features."""
    origin = 200
    before = _features_at(synthetic_arrays, origin)
    changed = synthetic_arrays.sales.copy()
    changed[:, origin - 1] += 50  # column origin-1 is day `origin`
    after = _features_at(replace(synthetic_arrays, sales=changed), origin)
    assert not before.equals(after)


def test_lag_7_definition(synthetic_arrays) -> None:
    """For h <= 7, lag_7 is exactly the sale 7 days before the target; it never reads past the origin."""
    origin = 150
    it, og, hs = grid(len(synthetic_arrays.items), [origin], 28)
    f = build_features(synthetic_arrays, it, og, hs)
    near = f[f["h"] <= 7]
    expected = synthetic_arrays.sales[near["item_idx"], near["day"] - 7 - 1]
    np.testing.assert_array_equal(near["lag_7"].to_numpy(), expected)
    lag_day = f["day"] - 7 * np.ceil(f["h"] / 7)
    assert (lag_day <= f["origin"]).all()


def test_training_rows_stop_at_fold_origin(synthetic_arrays) -> None:
    """No training target may fall inside (or after) the test window."""
    cfg = ModelConfig(first_train_origin=40, train_origin_stride=5)
    for origin in fold_origins(synthetic_arrays.last_observed_day, 3, cfg.horizon):
        rows = training_rows(synthetic_arrays, origin, cfg)
        assert rows["day"].max() <= origin
        assert rows["y"].notna().all()


def test_fold_origins_tile_the_end() -> None:
    assert fold_origins(1913, 4, 28) == [1801, 1829, 1857, 1885]
