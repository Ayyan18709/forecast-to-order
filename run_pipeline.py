"""Run the full Forecast to Order pipeline end to end.

Usage:
    python run_pipeline.py [--store CA_1] [--items 50] [--folds 4] [--refresh-data]

Steps: build/load the data panel -> rolling-origin backtest (baselines +
LightGBM) -> accuracy metrics -> inventory simulation for both ordering
policies -> sensitivity analysis. Everything is written to ./outputs/.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, replace

import numpy as np
import pandas as pd

from src.backtest import compute_metrics, fold_origins, run_backtest
from src.config import DEFAULT_CONFIG, Config
from src.data import load_panel, resolve_sales_file
from src.features import PanelArrays, panel_to_arrays
from src.simulation import (AverageBasedPolicy, ForecastBasedPolicy, SimulationInputs, assign_item_params,
                            make_policies, simulate, summarize)

log = logging.getLogger("pipeline")


def build_sim_inputs(arr: PanelArrays, daily: pd.DataFrame, start_origin: int, horizon: int) -> SimulationInputs:
    """Arrange actual demand and daily-origin forecasts for the simulation window.

    The window runs from ``start_origin + 1`` to the last observed day.
    Forecast row ``t`` was made at origin ``start_origin + t`` (the evening
    before simulation day ``t``).
    """
    end = arr.last_observed_day
    T = end - start_origin
    n = len(arr.items)
    fc = np.full((n, T, horizon, 4), np.nan, dtype=np.float32)
    t_idx = daily["origin"].to_numpy() - start_origin
    keep = (t_idx >= 0) & (t_idx < T)
    d = daily[keep]
    fc[d["item_idx"].to_numpy(), t_idx[keep], d["h"].to_numpy() - 1] = d[["lgbm", "p10", "p50", "p90"]].to_numpy()
    if np.isnan(fc).any():
        raise ValueError("Missing daily forecasts for the simulation window")
    return SimulationInputs(
        items=arr.items,
        demand=arr.sales[:, start_origin:end],
        history=arr.sales[:, :start_origin],
        forecasts=fc,
        prices=arr.price[:, start_origin:end].mean(axis=1),
        dates=arr.dates[start_origin:end],
    )


def save_sim_inputs(inputs: SimulationInputs, path) -> None:
    """Persist simulation inputs so the dashboard can rerun the simulation instantly."""
    np.savez_compressed(path, items=np.array(inputs.items), demand=inputs.demand, history=inputs.history,
                        forecasts=inputs.forecasts, prices=inputs.prices, dates=inputs.dates.astype("datetime64[D]"))


def load_sim_inputs(path) -> SimulationInputs:
    """Inverse of :func:`save_sim_inputs`."""
    z = np.load(path, allow_pickle=False)
    return SimulationInputs(list(z["items"]), z["demand"], z["history"], z["forecasts"], z["prices"], z["dates"])


def sensitivity(inputs: SimulationInputs, cfg: Config) -> pd.DataFrame:
    """Re-run the simulation across service levels, shelf lives and lead times."""
    rows = []
    scenarios = (
        [("service_level", v, replace(cfg.sim, service_level=v)) for v in (0.80, 0.90, 0.95, 0.98, 0.99)]
        + [("shelf_life_multiplier", v, replace(cfg.sim, shelf_life_multiplier=v)) for v in (0.5, 0.75, 1.0, 1.5, 2.0)]
        + [("lead_time", v, replace(cfg.sim, lead_time=v)) for v in (2, 7, 14)]
    )
    for name, value, sim_cfg in scenarios:
        policies = make_policies(sim_cfg)
        policies["Forecast based (no expiry cap)"] = ForecastBasedPolicy(sim_cfg.service_level, sim_cfg.review_period, False)
        s = summarize(simulate(inputs, sim_cfg, policies))
        s.insert(0, "value", value)
        s.insert(0, "parameter", name)
        rows.append(s)
    return pd.concat(rows, ignore_index=True)


def frontier(inputs: SimulationInputs, cfg: Config) -> pd.DataFrame:
    """Trade-off curve: cash tied up vs lost sales as each policy's safety knob is turned.

    The average rule is tuned by its extra days of cover, the forecast rule by
    its service level. Comparing curves (not single points) is the fair test.
    """
    sim = cfg.sim
    policies: dict[str, object] = {}
    for d in (0, 3, 7, 10, 14, 21):
        policies[f"Average based|{d} extra days"] = AverageBasedPolicy(sim.lead_time + sim.review_period + d, sim.review_period)
    for sl in (0.80, 0.90, 0.95, 0.98, 0.99, 0.995):
        policies[f"Forecast based|{sl:.1%} service"] = ForecastBasedPolicy(sl, sim.review_period, sim.expiry_aware)
    s = summarize(simulate(inputs, sim, policies))
    s[["policy", "setting"]] = s["policy"].str.split("|", expand=True)
    return s


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", default=DEFAULT_CONFIG.data.store_id)
    parser.add_argument("--items", type=int, default=DEFAULT_CONFIG.data.n_items)
    parser.add_argument("--folds", type=int, default=DEFAULT_CONFIG.model.n_folds)
    parser.add_argument("--refresh-data", action="store_true", help="rebuild the cached parquet panel")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    cfg = replace(DEFAULT_CONFIG,
                  data=replace(DEFAULT_CONFIG.data, store_id=args.store, n_items=args.items),
                  model=replace(DEFAULT_CONFIG.model, n_folds=args.folds))
    out = cfg.outputs_dir
    out.mkdir(exist_ok=True)
    t0 = time.perf_counter()

    log.info("1/5 Data: %s, store %s, top %d items", resolve_sales_file(cfg.data).name, cfg.data.store_id, cfg.data.n_items)
    panel = load_panel(cfg.data, cfg.model.horizon, refresh=args.refresh_data)
    arr = panel_to_arrays(panel)
    item_dates = dict(zip(range(len(arr.dates)), arr.dates))

    log.info("2/5 Backtest: %d folds x %d days", cfg.model.n_folds, cfg.model.horizon)
    bt = run_backtest(arr, cfg.model)
    for df in (bt.fold_forecasts, bt.daily_forecasts):
        df.insert(0, "item_id", [arr.items[i] for i in df["item_idx"]])
    bt.fold_forecasts["date"] = [item_dates[d - 1] for d in bt.fold_forecasts["day"]]
    bt.fold_forecasts.to_parquet(out / "forecasts_backtest.parquet", index=False)
    bt.daily_forecasts.to_parquet(out / "forecasts_daily.parquet", index=False)
    bt.folds.to_csv(out / "folds.csv", index=False)
    imp = bt.importance.groupby("feature")["gain"].mean().sort_values(ascending=False)
    imp.to_csv(out / "feature_importance.csv")

    log.info("3/5 Metrics")
    overall, per_fold, per_item = compute_metrics(bt.fold_forecasts, arr.items)
    overall.to_csv(out / "metrics_overall.csv", index=False)
    per_fold.to_csv(out / "metrics_per_fold.csv")
    per_item.to_csv(out / "metrics_per_item.csv", index=False)

    log.info("4/5 Inventory simulation")
    start = fold_origins(arr.last_observed_day, cfg.model.n_folds, cfg.model.horizon)[0]
    inputs = build_sim_inputs(arr, bt.daily_forecasts, start, cfg.model.horizon)
    save_sim_inputs(inputs, out / "sim_inputs.npz")
    params = assign_item_params(inputs.items, inputs.prices, cfg.sim)
    pd.DataFrame([asdict(p) for p in params]).to_csv(out / "item_params.csv", index=False)
    sim_log = simulate(inputs, cfg.sim)
    sim_log.to_parquet(out / "simulation_daily.parquet", index=False)
    summary = summarize(sim_log)
    summary.to_csv(out / "simulation_summary.csv", index=False)
    summarize(sim_log, by=["item_id"]).to_csv(out / "simulation_per_item.csv", index=False)

    log.info("5/5 Sensitivity analysis")
    sensitivity(inputs, cfg).to_csv(out / "sensitivity.csv", index=False)
    frontier(inputs, cfg).to_csv(out / "frontier.csv", index=False)

    info = {
        "sales_file": resolve_sales_file(cfg.data).name,
        "store_id": cfg.data.store_id, "n_items": len(arr.items),
        "last_observed_date": str(arr.dates[arr.last_observed_day - 1])[:10],
        "simulation_start": str(inputs.dates[0])[:10], "simulation_end": str(inputs.dates[-1])[:10],
        "simulation_days": int(inputs.demand.shape[1]),
        "config": {"model": {k: v for k, v in asdict(cfg.model).items()}, "sim": asdict(cfg.sim)},
        "runtime_seconds": round(time.perf_counter() - t0, 1),
    }
    (out / "run_info.json").write_text(json.dumps(info, indent=2, default=str))

    pd.set_option("display.width", 160)
    print("\n=== Forecast accuracy (all folds) ===")
    print(overall.T.round(3).to_string())
    print("\n=== Inventory simulation ===")
    print(summary.set_index("policy").T.round(2).to_string())
    print(f"\nDone in {info['runtime_seconds']}s. Outputs in {out}")


if __name__ == "__main__":
    main()
