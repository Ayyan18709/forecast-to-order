"""Forecasting models: two business-style baselines and a global LightGBM.

The baselines need no training and are computed from the same leakage-safe
features the main model sees, so all methods use identical information.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features import CATEGORICAL, FEATURES

log = logging.getLogger(__name__)


def moving_average_forecast(feats: pd.DataFrame) -> np.ndarray:
    """Baseline 1: flat forecast equal to the 28-day average at the origin."""
    return feats["roll_mean_28"].to_numpy()


def seasonal_naive_forecast(feats: pd.DataFrame) -> np.ndarray:
    """Baseline 2: same weekday last week (repeated weekly across the horizon)."""
    return feats["lag_7"].fillna(feats["roll_mean_28"]).to_numpy()


@dataclass
class GlobalLGBM:
    """One LightGBM model across all items for the mean, plus quantile models.

    The mean model uses a Tweedie objective, which suits non-negative,
    zero-inflated count demand. Quantile models give the P10 / P50 / P90 band.
    """

    params: dict
    num_boost_round: int = 300
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    seed: int = 42
    models: dict[str, lgb.Booster] = field(default_factory=dict)

    def _train(self, X: pd.DataFrame, y: np.ndarray, objective: dict) -> lgb.Booster:
        params = {**self.params, **objective, "seed": self.seed, "deterministic": True, "num_threads": 0}
        ds = lgb.Dataset(X, label=y, categorical_feature=CATEGORICAL, free_raw_data=False)
        return lgb.train(params, ds, num_boost_round=self.num_boost_round)

    def fit(self, train: pd.DataFrame) -> "GlobalLGBM":
        """Fit the mean and quantile models on rows with a known target."""
        train = train[train["y"].notna()]
        X, y = train[FEATURES], train["y"].to_numpy()
        self.models["mean"] = self._train(X, y, {"objective": "tweedie", "tweedie_variance_power": 1.2})
        for q in self.quantiles:
            self.models[f"p{round(q * 100)}"] = self._train(X, y, {"objective": "quantile", "alpha": q})
        return self

    def predict(self, feats: pd.DataFrame) -> pd.DataFrame:
        """Predict mean and quantiles; quantiles are clipped at 0 and sorted."""
        X = feats[FEATURES]
        out = pd.DataFrame(index=feats.index)
        out["mean"] = np.clip(self.models["mean"].predict(X), 0, None)
        q_names = [f"p{round(q * 100)}" for q in self.quantiles]
        q = np.column_stack([np.clip(self.models[n].predict(X), 0, None) for n in q_names])
        q.sort(axis=1)  # guard against quantile crossing
        for i, n in enumerate(q_names):
            out[n] = q[:, i]
        return out

    def feature_importance(self) -> pd.DataFrame:
        """Gain-based importance of the mean model, normalised to sum to 1."""
        booster = self.models["mean"]
        imp = pd.DataFrame({"feature": booster.feature_name(), "gain": booster.feature_importance("gain")})
        imp["gain"] /= imp["gain"].sum()
        return imp.sort_values("gain", ascending=False).reset_index(drop=True)
