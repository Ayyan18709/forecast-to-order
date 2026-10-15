"""Rolling-origin backtest.

For each fold we pick an origin ``O`` (the last day treated as known), train
only on rows whose target day is ``<= O``, and forecast days ``O+1 .. O+28``.
Folds step forward 28 days so the test windows tile the final months of data.

Two kinds of forecasts are produced per fold:

* **fold forecasts** - one forecast from origin ``O`` for the full 28 days;
  used for accuracy metrics and the dashboard chart.
* **daily forecasts** - a fresh 28-day forecast from every origin
  ``O .. O+27`` (as if re-forecasting each morning) using the same fold
  model; used by the inventory simulation so each ordering decision on day
  ``t`` only sees a forecast made at ``t-1``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import ModelConfig
from src.features import PanelArrays, build_features, grid
from src.models import GlobalLGBM, moving_average_forecast, seasonal_naive_forecast

log = logging.getLogger(__name__)

MODEL_COLUMNS = {
    "ma28": "28-day moving average",
    "snaive": "Seasonal naive (last week)",
    "lgbm": "LightGBM (mean)",
    "p50": "LightGBM (P50)",
}


@dataclass
class BacktestResult:
    """Outputs of :func:`run_backtest`."""

    fold_forecasts: pd.DataFrame
    daily_forecasts: pd.DataFrame
    importance: pd.DataFrame
    folds: pd.DataFrame


def fold_origins(last_day: int, n_folds: int, horizon: int) -> list[int]:
    """Origins of the rolling folds, oldest first, the last ending at ``last_day``."""
    return [last_day - horizon * (n_folds - k) for k in range(n_folds)]


def training_rows(arr: PanelArrays, origin: int, cfg: ModelConfig) -> pd.DataFrame:
    """Training rows for a fold: every target day is ``<= origin``."""
    origins = np.arange(origin - 1, cfg.first_train_origin - 1, -cfg.train_origin_stride)[::-1]
    items, orgs, hs = grid(len(arr.items), origins, cfg.horizon)
    keep = orgs + hs <= origin
    feats = build_features(arr, items[keep], orgs[keep], hs[keep])
    assert feats["day"].max() <= origin, "training target leaks into the test window"
    return feats


def _with_baselines(feats: pd.DataFrame, model: GlobalLGBM) -> pd.DataFrame:
    preds = model.predict(feats)
    out = feats[["item_idx", "origin", "h", "day", "y"]].copy()
    out["ma28"] = moving_average_forecast(feats)
    out["snaive"] = seasonal_naive_forecast(feats)
    out["lgbm"] = preds["mean"].to_numpy()
    for c in ("p10", "p50", "p90"):
        out[c] = preds[c].to_numpy()
    return out


def run_backtest(arr: PanelArrays, cfg: ModelConfig) -> BacktestResult:
    """Run all folds and collect forecasts, importances and fold metadata."""
    origins = fold_origins(arr.last_observed_day, cfg.n_folds, cfg.horizon)
    n_items = len(arr.items)
    fold_fc, daily_fc, imps, meta = [], [], [], []
    for k, origin in enumerate(origins):
        t0 = time.perf_counter()
        train = training_rows(arr, origin, cfg)
        model = GlobalLGBM(cfg.lgb_params, cfg.num_boost_round, cfg.quantiles, cfg.seed).fit(train)

        it, og, hs = grid(n_items, [origin], cfg.horizon)
        fc = _with_baselines(build_features(arr, it, og, hs), model)
        fc["fold"] = k
        fold_fc.append(fc)

        it, og, hs = grid(n_items, np.arange(origin, origin + cfg.horizon), cfg.horizon)
        dfc = _with_baselines(build_features(arr, it, og, hs), model)
        dfc["fold"] = k
        daily_fc.append(dfc.drop(columns=["snaive"]))

        imp = model.feature_importance()
        imp["fold"] = k
        imps.append(imp)
        meta.append({"fold": k, "origin": origin, "train_rows": len(train),
                     "train_last_target_day": int(train["day"].max()),
                     "test_first_day": origin + 1, "test_last_day": origin + cfg.horizon,
                     "seconds": round(time.perf_counter() - t0, 1)})
        log.info("Fold %d: origin d_%d, %d training rows, %.1fs", k, origin, len(train), meta[-1]["seconds"])

    return BacktestResult(pd.concat(fold_fc, ignore_index=True), pd.concat(daily_fc, ignore_index=True),
                          pd.concat(imps, ignore_index=True), pd.DataFrame(meta))


def _metrics(df: pd.DataFrame) -> pd.Series:
    y = df["y"].to_numpy()
    total = y.sum()
    out: dict[str, float] = {"actual_units": total}
    for col in MODEL_COLUMNS:
        err = df[col].to_numpy() - y
        out[f"wape_{col}"] = np.abs(err).sum() / total if total else np.nan
        out[f"bias_{col}"] = err.sum() / total if total else np.nan
    out["coverage_p10_p90"] = ((y >= df["p10"]) & (y <= df["p90"])).mean()
    return pd.Series(out)


def compute_metrics(fold_forecasts: pd.DataFrame, items: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """WAPE, bias and P10-P90 coverage: overall, per fold and per item.

    WAPE = sum|forecast - actual| / sum(actual); bias = sum(forecast - actual) / sum(actual)
    (positive = over-forecasting).
    """
    df = fold_forecasts[fold_forecasts["y"].notna()]
    overall = _metrics(df).to_frame("overall").T
    per_fold = df.groupby("fold").apply(_metrics, include_groups=False)
    per_item = df.groupby("item_idx").apply(_metrics, include_groups=False)
    per_item.insert(0, "item_id", [items[i] for i in per_item.index])
    return overall, per_fold, per_item.reset_index(drop=True)
