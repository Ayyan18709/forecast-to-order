"""Sanity checks for the FEFO inventory environment and ordering policies."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.simulation import (AverageBasedPolicy, ForecastBasedPolicy, InventoryEnv, ItemParams, _norm_ppf,
                            run_episode)


def _params(shelf_life: int = 30) -> ItemParams:
    return ItemParams("X", shelf_life, price=10.0, unit_cost=7.0, holding_cost_per_day=0.01)


class FixedOrders:
    """Policy placing pre-set orders on given days."""

    def __init__(self, orders: dict[int, float]) -> None:
        self.orders = orders

    def __call__(self, obs: dict) -> float:
        return self.orders.get(obs["t"], 0.0)


def _random_run(seed: int) -> tuple[pd.DataFrame, InventoryEnv]:
    rng = np.random.default_rng(seed)
    demand = rng.poisson(rng.uniform(0.5, 15), size=120).astype(float)
    env = InventoryEnv(demand, rng.poisson(5, 60).astype(float), _params(int(rng.integers(2, 40))),
                       lead_time=int(rng.integers(1, 10)), initial_stock=float(rng.integers(0, 100)))
    orders = {t: float(rng.integers(0, 120)) for t in range(0, 120, 7)}
    return run_episode(env, FixedOrders(orders)), env


@pytest.mark.parametrize("seed", range(20))
def test_inventory_never_negative_and_units_balance(seed: int) -> None:
    log, env = _random_run(seed)
    assert (log["on_hand"] >= -1e-9).all()
    assert (log["sold"] <= log["demand"] + 1e-9).all()
    assert np.allclose(log["sold"] + log["lost"], log["demand"])
    # Conservation: start + received = sold + expired + end stock
    assert np.isclose(env.initial_stock + log["received"].sum(),
                      log["sold"].sum() + log["expired"].sum() + log["on_hand"].iloc[-1])


def test_expired_units_are_never_sold() -> None:
    """Stock with a 3-day shelf life that nobody buys in time is discarded, not sold later."""
    demand = np.array([0, 0, 0, 50, 0], dtype=float)
    env = InventoryEnv(demand, np.zeros(28), _params(shelf_life=3), lead_time=1, initial_stock=20)
    log = run_episode(env, FixedOrders({}))
    assert log.loc[3, "expired"] == 20  # expires at the start of day 3
    assert log["sold"].sum() == 0
    assert log.loc[3, "lost"] == 50


def test_fefo_sells_oldest_first() -> None:
    """With an old and a new batch on hand, sales come out of the older batch."""
    demand = np.array([0, 0, 0, 10, 0, 0, 0, 0, 0, 0], dtype=float)
    env = InventoryEnv(demand, np.zeros(28), _params(shelf_life=6), lead_time=2, initial_stock=10)
    log = run_episode(env, FixedOrders({0: 10}))  # second batch arrives day 2, expires day 8
    # The old batch (expires day 6) is sold on day 3, so nothing expires on day 6.
    assert log.loc[6, "expired"] == 0
    assert log.loc[8, "expired"] == 10


def test_orders_arrive_after_lead_time() -> None:
    env = InventoryEnv(np.zeros(15), np.zeros(28), _params(), lead_time=5, initial_stock=0)
    log = run_episode(env, FixedOrders({2: 40}))
    assert log.loc[7, "received"] == 40
    assert log["received"].sum() == 40
    assert log.loc[6, "on_hand"] == 0 and log.loc[7, "on_hand"] == 40


def test_average_policy_only_orders_on_review_days() -> None:
    env = InventoryEnv(np.full(30, 4.0), np.full(28, 4.0), _params(), lead_time=3, initial_stock=0)
    log = run_episode(env, AverageBasedPolicy(coverage_days=10, review_period=7))
    assert (log.loc[log["t"] % 7 != 0, "order"] == 0).all()
    assert log.loc[0, "order"] == 40


def test_forecast_policy_with_perfect_forecast_avoids_stockouts() -> None:
    """Constant demand + exact zero-uncertainty forecast => no lost sales, no expiry, bounded stock."""
    T, H = 60, 28
    fc = np.full((T, H, 4), 6.0)
    env = InventoryEnv(np.full(T, 6.0), np.full(28, 6.0), _params(90), lead_time=4, initial_stock=6 * 11, forecasts=fc)
    log = run_episode(env, ForecastBasedPolicy(service_level=0.95, review_period=7))
    assert log["lost"].sum() == 0
    assert log["expired"].sum() == 0
    assert log["on_hand"].max() <= 6 * 12


def test_expiry_cap_limits_orders_for_short_shelf_life() -> None:
    fc = np.full((28, 4), 2.0)
    fc[:, 1], fc[:, 3] = 0.0, 6.0  # wide uncertainty band
    obs = {"t": 0, "forecast": fc, "position": 0.0, "lead_time": 7, "params": _params(shelf_life=2)}
    capped = ForecastBasedPolicy(0.99, 7, expiry_aware=True)(obs)
    uncapped = ForecastBasedPolicy(0.99, 7, expiry_aware=False)(obs)
    assert capped == 2.0 * (7 + 2)
    assert uncapped > capped


def test_norm_ppf_matches_known_values() -> None:
    for p, z in [(0.5, 0.0), (0.9, 1.2815515655), (0.95, 1.6448536270), (0.99, 2.3263478740), (0.01, -2.3263478740)]:
        assert abs(_norm_ppf(p) - z) < 1e-8
