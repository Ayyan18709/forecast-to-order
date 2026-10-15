"""Leakage-safe feature construction for direct multi-horizon forecasting.

Every training or prediction row is defined by three things:

* ``item``   - which product,
* ``origin`` - the last day whose sales are known when the forecast is made,
* ``h``      - how many days ahead we forecast (target day ``t = origin + h``).

All sales-based features are computed from days ``<= origin`` only. Calendar
fields and prices describe the target day; they are known in advance (store
calendars and planned prices), which is the standard M5 assumption.

Lag features keep their usual names but are anchored to the origin:
``lag_7`` is the sales on the most recent day with the same weekday as the
target that is not after the origin (exactly ``t - 7`` when ``h <= 7``,
``t - 14`` when ``8 <= h <= 14``, and so on). ``lag_14`` and ``lag_28`` step
back a further 1 and 3 weeks.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURES: list[str] = [
    "item_code", "dept_code", "h",
    "wday", "month", "mday", "snap", "event_type", "is_event", "event_next_3d",
    "lag_7", "lag_14", "lag_28", "last_sales", "wd_mean_4",
    "roll_mean_7", "roll_mean_14", "roll_mean_28", "roll_mean_56", "roll_mean_182",
    "roll_std_7", "roll_std_28", "zero_frac_28",
    "price", "price_vs_origin", "price_wow", "price_rel_182",
]
CATEGORICAL: list[str] = ["item_code", "dept_code", "event_type"]


@dataclass
class PanelArrays:
    """Dense matrices extracted from the long panel (rows = items, cols = days).

    Column ``j`` corresponds to day ``j + 1`` (M5 days are 1-based).
    """

    items: list[str]
    depts: np.ndarray  # dept code per item
    sales: np.ndarray  # float64 [n_items, n_days], NaN for unknown future days
    price: np.ndarray  # float64 [n_items, n_days]
    wday: np.ndarray  # int [n_days]
    month: np.ndarray
    mday: np.ndarray
    snap: np.ndarray
    event_type: np.ndarray  # int code, 0 = no event
    dates: np.ndarray
    last_observed_day: int

    @property
    def n_days(self) -> int:
        """Number of days covered (observed + future)."""
        return self.sales.shape[1]


def panel_to_arrays(panel: pd.DataFrame) -> PanelArrays:
    """Convert the long panel produced by :mod:`src.data` into matrices."""
    items = list(panel["item_id"].cat.categories[panel["item_id"].cat.codes.unique()])
    items = sorted(items)
    p = panel.copy()
    p["item_id"] = p["item_id"].astype(str)
    sales = p.pivot(index="item_id", columns="day", values="sales").loc[items].to_numpy(dtype=float)
    price = p.pivot(index="item_id", columns="day", values="sell_price").loc[items].to_numpy(dtype=float)
    depts_s = p.drop_duplicates("item_id").set_index("item_id").loc[items, "dept_id"].astype(str)
    depts = pd.Categorical(depts_s).codes.astype(int)

    cal = p.drop_duplicates("day").sort_values("day")
    if not np.array_equal(cal["day"].to_numpy(), np.arange(1, len(cal) + 1)):
        raise ValueError("Panel days must be contiguous and start at 1")
    ev = cal["event_type_1"].astype(str).replace({"nan": None})
    ev_codes = {"Cultural": 1, "National": 2, "Religious": 3, "Sporting": 4}
    event_type = ev.map(lambda x: ev_codes.get(x, 0) if x else 0).to_numpy(dtype=int)
    return PanelArrays(
        items=items,
        depts=depts,
        sales=sales,
        price=price,
        wday=cal["wday"].to_numpy(dtype=int),
        month=cal["month"].to_numpy(dtype=int),
        mday=cal["date"].dt.day.to_numpy(dtype=int),
        snap=cal["snap"].to_numpy(dtype=int),
        event_type=event_type,
        dates=cal["date"].to_numpy(),
        last_observed_day=int(panel.attrs.get("last_observed_day", np.where(~np.isnan(sales).all(0))[0].max() + 1)),
    )


def _cumsums(m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prefix sums of values, squares and zero indicators, with a leading zero column.

    Prefix sums are causal: entry ``k`` only depends on columns ``< k``, so
    reading them at an origin never touches later (unknown) days.
    """
    x = np.nan_to_num(m, nan=0.0)
    pad = np.zeros((m.shape[0], 1))
    cs = np.hstack([pad, np.cumsum(x, axis=1)])
    cs2 = np.hstack([pad, np.cumsum(x * x, axis=1)])
    cz = np.hstack([pad, np.cumsum((m == 0).astype(float), axis=1)])
    return cs, cs2, cz


def _window_stats(prefix: tuple[np.ndarray, np.ndarray, np.ndarray], end_idx: np.ndarray, items: np.ndarray,
                  window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean, std and zero-share over ``window`` days ending at 0-based column ``end_idx`` (inclusive).

    Windows reaching before day 1 are truncated to the available history.
    """
    cs, cs2, cz = prefix
    hi = end_idx + 1
    lo = np.clip(hi - window, 0, None)
    n = (hi - lo).astype(float)
    s = cs[items, hi] - cs[items, lo]
    s2 = cs2[items, hi] - cs2[items, lo]
    mean = s / n
    std = np.sqrt(np.clip(s2 / n - mean**2, 0, None))
    zero = (cz[items, hi] - cz[items, lo]) / n
    return mean, std, zero


def build_features(arr: PanelArrays, item_idx: np.ndarray, origins: np.ndarray, horizons: np.ndarray) -> pd.DataFrame:
    """Build the feature matrix for the given (item, origin, horizon) rows.

    Args:
        arr: Dense panel matrices.
        item_idx: Row index into ``arr.items`` for each sample.
        origins: Forecast origin (1-based day) for each sample. Only sales on
            days ``<= origin`` are read.
        horizons: Days ahead (1..H) for each sample.

    Returns:
        DataFrame with :data:`FEATURES` plus ``item_idx``, ``origin``, ``h``,
        ``day`` (target day) and ``y`` (actual sales, NaN if unknown).
    """
    item_idx = np.asarray(item_idx, dtype=int)
    origins = np.asarray(origins, dtype=int)
    horizons = np.asarray(horizons, dtype=int)
    target = origins + horizons  # 1-based target day
    o_col = origins - 1  # 0-based column of origin
    t_col = target - 1
    if (t_col >= arr.n_days).any():
        raise ValueError("Target day beyond panel range")

    # Only this view of sales is used below: everything after the origin is hidden.
    weeks_back = -(-horizons // 7)  # ceil(h / 7)
    def lag_at(extra_weeks: int) -> np.ndarray:
        col = t_col - 7 * (weeks_back + extra_weeks)
        assert (col <= o_col).all()
        out = arr.sales[item_idx, np.clip(col, 0, None)]
        return np.where(col >= 0, out, np.nan)

    f: dict[str, np.ndarray] = {}
    f["item_code"] = item_idx
    f["dept_code"] = arr.depts[item_idx]
    f["h"] = horizons
    f["wday"] = arr.wday[t_col]
    f["month"] = arr.month[t_col]
    f["mday"] = arr.mday[t_col]
    f["snap"] = arr.snap[t_col]
    f["event_type"] = arr.event_type[t_col]
    f["is_event"] = (arr.event_type[t_col] > 0).astype(int)
    ev_ahead = np.zeros(len(t_col), dtype=int)
    for k in (1, 2, 3):
        ev_ahead |= arr.event_type[np.clip(t_col + k, None, arr.n_days - 1)] > 0
    f["event_next_3d"] = ev_ahead

    l1, l2, l3, l4 = (lag_at(k) for k in range(4))
    f["lag_7"], f["lag_14"], f["lag_28"] = l1, l2, l4
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN at the very start of history
        f["wd_mean_4"] = np.nanmean(np.vstack([l1, l2, l3, l4]), axis=0)
    f["last_sales"] = arr.sales[item_idx, o_col]

    sales_prefix = _cumsums(arr.sales)
    for w in (7, 14, 28, 56, 182):
        mean, std, zero = _window_stats(sales_prefix, o_col, item_idx, w)
        f[f"roll_mean_{w}"] = mean
        if w in (7, 28):
            f[f"roll_std_{w}"] = std
        if w == 28:
            f["zero_frac_28"] = zero

    price_t = arr.price[item_idx, t_col]
    price_o = arr.price[item_idx, o_col]
    price_wk = arr.price[item_idx, np.clip(t_col - 7, 0, None)]
    p_mean, _, _ = _window_stats(_cumsums(arr.price), o_col, item_idx, 182)
    f["price"] = price_t
    f["price_vs_origin"] = price_t / price_o - 1
    f["price_wow"] = price_t / price_wk - 1
    f["price_rel_182"] = price_t / p_mean - 1

    df = pd.DataFrame(f)
    df["item_idx"] = item_idx
    df["origin"] = origins
    df["day"] = target
    df["y"] = arr.sales[item_idx, t_col]
    return df


def grid(n_items: int, origins: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All (item, origin, h) combinations as three aligned arrays."""
    items, orgs, hs = np.meshgrid(np.arange(n_items), np.asarray(origins), np.arange(1, horizon + 1), indexing="ij")
    return items.ravel(), orgs.ravel(), hs.ravel()
