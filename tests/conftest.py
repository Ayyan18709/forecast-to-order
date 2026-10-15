"""Shared fixtures: a small synthetic panel so tests run without the M5 files."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features import PanelArrays  # noqa: E402


@pytest.fixture
def synthetic_arrays() -> PanelArrays:
    """Three items over 400 days with weekly seasonality, events and price changes."""
    rng = np.random.default_rng(0)
    n_items, n_days = 3, 400
    days = np.arange(n_days)
    base = np.array([5.0, 20.0, 1.0])[:, None]
    weekly = 1 + 0.3 * np.sin(2 * np.pi * days / 7)[None, :]
    sales = rng.poisson(base * weekly).astype(float)
    price = 2.0 + 0.5 * ((rng.random((n_items, n_days)) < 0.05).cumsum(axis=1) % 3)
    event = np.zeros(n_days, dtype=int)
    event[::37] = 2
    return PanelArrays(
        items=["A", "B", "C"],
        depts=np.array([0, 0, 1]),
        sales=sales,
        price=price,
        wday=(days % 7) + 1,
        month=(days // 30) % 12 + 1,
        mday=days % 28 + 1,
        snap=(days % 30 < 10).astype(int),
        event_type=event,
        dates=np.datetime64("2020-01-01") + days.astype("timedelta64[D]"),
        last_observed_day=n_days - 28,
    )
