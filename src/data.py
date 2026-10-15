"""Data pipeline: load raw M5 files, slice to one store and its top items, and
turn the wide sales table into a tidy daily panel enriched with calendar and
price information. The result is cached as parquet.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DataConfig

log = logging.getLogger(__name__)

ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def resolve_sales_file(cfg: DataConfig) -> Path:
    """Return the first sales file from ``cfg.sales_files`` that exists."""
    for name in cfg.sales_files:
        path = cfg.data_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"None of {cfg.sales_files} found in {cfg.data_dir}")


def load_store_sales(cfg: DataConfig) -> pd.DataFrame:
    """Load the wide sales table, keeping only rows for ``cfg.store_id``.

    The file is read in chunks and filtered before anything else so the full
    30k x 1.9k table is never held in memory at once. Day columns are int16.
    """
    path = resolve_sales_file(cfg)
    header = pd.read_csv(path, nrows=0).columns
    day_cols = [c for c in header if c.startswith("d_")]
    dtypes = {c: "int16" for c in day_cols}
    dtypes.update({c: "string" for c in ID_COLS})
    chunks = [
        chunk[chunk["store_id"] == cfg.store_id]
        for chunk in pd.read_csv(path, dtype=dtypes, chunksize=5000)
    ]
    wide = pd.concat(chunks, ignore_index=True)
    if wide.empty:
        raise ValueError(f"Store {cfg.store_id!r} not found in {path.name}")
    log.info("Loaded %s rows for %s from %s (%d days)", len(wide), cfg.store_id, path.name, len(day_cols))
    return wide


def top_items(wide: pd.DataFrame, n_items: int) -> list[str]:
    """Item ids of the ``n_items`` best sellers by total units."""
    day_cols = [c for c in wide.columns if c.startswith("d_")]
    totals = wide.set_index("item_id")[day_cols].astype("int64").sum(axis=1)
    return totals.sort_values(ascending=False).head(n_items).index.tolist()


def load_calendar(cfg: DataConfig) -> pd.DataFrame:
    """Calendar with a numeric ``day`` index and the store's SNAP flag."""
    cal = pd.read_csv(cfg.data_dir / cfg.calendar_file, parse_dates=["date"])
    cal["day"] = cal["d"].str[2:].astype("int16")
    cal["snap"] = cal[f"snap_{cfg.state_id}"].astype("int8")
    keep = ["day", "date", "wm_yr_wk", "wday", "month", "year",
            "event_name_1", "event_type_1", "event_name_2", "event_type_2", "snap"]
    return cal[keep]


def load_prices(cfg: DataConfig, items: list[str]) -> pd.DataFrame:
    """Weekly prices for the chosen store and items."""
    prices = pd.read_csv(
        cfg.data_dir / cfg.prices_file,
        dtype={"store_id": "string", "item_id": "string", "wm_yr_wk": "int32", "sell_price": "float32"},
    )
    prices = prices[(prices["store_id"] == cfg.store_id) & prices["item_id"].isin(items)]
    return prices.drop(columns="store_id")


def build_panel(cfg: DataConfig, horizon: int = 28) -> pd.DataFrame:
    """Build the tidy item x day panel.

    The panel covers every observed day plus ``horizon`` future days (sales are
    NaN there) so that calendar and planned-price features exist for the
    final forecast window.
    """
    wide = load_store_sales(cfg)
    items = top_items(wide, cfg.n_items)
    wide = wide[wide["item_id"].isin(items)]
    day_cols = [c for c in wide.columns if c.startswith("d_")]
    last_day = int(day_cols[-1][2:])

    long = wide.melt(id_vars=["item_id", "dept_id"], value_vars=day_cols, var_name="d", value_name="sales")
    long["day"] = long["d"].str[2:].astype("int16")
    long = long.drop(columns="d")

    # Future rows: sales unknown, but calendar and prices are available.
    future_days = np.arange(last_day + 1, last_day + horizon + 1, dtype="int16")
    item_dept = wide[["item_id", "dept_id"]].drop_duplicates()
    future = item_dept.merge(pd.DataFrame({"day": future_days}), how="cross")
    long = pd.concat([long, future], ignore_index=True)
    long["sales"] = long["sales"].astype("float32")

    cal = load_calendar(cfg)
    long = long.merge(cal, on="day", how="left")
    prices = load_prices(cfg, items)
    long = long.merge(prices, on=["item_id", "wm_yr_wk"], how="left")

    long = long.sort_values(["item_id", "day"]).reset_index(drop=True)
    # Prices are missing before an item's launch; fill so features stay defined.
    long["sell_price"] = long.groupby("item_id")["sell_price"].transform(lambda s: s.ffill().bfill())
    for col in ["item_id", "dept_id", "event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
        long[col] = long[col].astype("category")
    long.attrs["last_observed_day"] = last_day
    return long


def load_panel(cfg: DataConfig, horizon: int = 28, refresh: bool = False) -> pd.DataFrame:
    """Load the cached panel, building and caching it if needed."""
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    cache = cfg.processed_dir / f"panel_{cfg.store_id}_top{cfg.n_items}.parquet"
    if cache.exists() and not refresh:
        panel = pd.read_parquet(cache)
    else:
        panel = build_panel(cfg, horizon)
        panel.to_parquet(cache, index=False)
        log.info("Cached panel to %s", cache)
    panel.attrs["last_observed_day"] = int(panel.loc[panel["sales"].notna(), "day"].max())
    return panel
