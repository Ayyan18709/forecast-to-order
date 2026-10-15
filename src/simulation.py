"""Expiry-aware inventory simulation.

:class:`InventoryEnv` simulates one item day by day with FEFO batches
(first-expired, first-out). It follows the familiar ``reset() / step(action)``
interface so a reinforcement learning agent can be plugged in later: the
action is the order quantity, the reward is minus the day's cost.

Order of events on day ``t``:

1. The order chosen for day ``t`` is placed; it arrives at the start of day ``t + lead_time``.
2. Orders due today are received as a new batch expiring on ``t + shelf_life``.
3. Batches whose expiry day is ``<= t`` are thrown away (counted as expired).
4. Today's demand is served from the soonest-expiring batch first; unmet demand is lost.
5. Holding cost is charged on the stock left at the end of the day.

The policy deciding the order for day ``t`` only sees demand up to ``t - 1``
and a forecast made at origin ``t - 1``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import SimConfig

Z_80 = 1.2815515655446004  # standard normal quantile at 0.9 (P10-P90 half-width)
FC_MEAN, FC_P10, FC_P50, FC_P90 = range(4)


@dataclass(frozen=True)
class ItemParams:
    """Economics of one item."""

    item_id: str
    shelf_life: int  # days a received batch can be sold
    price: float  # selling price per unit
    unit_cost: float  # purchase cost per unit
    holding_cost_per_day: float  # cost of keeping one unit for one day

    @property
    def margin(self) -> float:
        """Profit lost per unit of unmet demand."""
        return self.price - self.unit_cost


def assign_item_params(items: list[str], prices: np.ndarray, cfg: SimConfig) -> list[ItemParams]:
    """Give each item a (simulated) shelf life and costs.

    Shelf lives are drawn from ``cfg.shelf_life_choices`` with a fixed seed so
    every run is identical, then scaled by ``cfg.shelf_life_multiplier``.
    """
    rng = np.random.default_rng(cfg.seed)
    base = rng.choice(cfg.shelf_life_choices, size=len(items))
    out = []
    for item, sl, price in zip(items, base, prices):
        cost = cfg.cost_ratio * float(price)
        out.append(ItemParams(
            item_id=item,
            shelf_life=max(1, int(round(sl * cfg.shelf_life_multiplier))),
            price=float(price),
            unit_cost=cost,
            holding_cost_per_day=cost * cfg.annual_holding_rate / 365.0,
        ))
    return out


@dataclass
class Batch:
    """Units that arrived together and share an expiry day."""

    expiry_day: int
    qty: float


@dataclass
class InventoryEnv:
    """Daily single-item inventory environment with FEFO batches and lead time.

    Args:
        demand: Actual daily demand over the simulation period (length ``T``).
        history: Demand before the simulation starts (used by policies for averages).
        params: Item economics.
        lead_time: Days between ordering and receiving (>= 1).
        initial_stock: Units on hand at the start, all fresh.
        forecasts: Optional array ``[T, H, 4]`` (mean, p10, p50, p90); row ``t``
            is the forecast made at the end of day ``t - 1`` for days ``t .. t+H-1``.
    """

    demand: np.ndarray
    history: np.ndarray
    params: ItemParams
    lead_time: int
    initial_stock: float
    forecasts: np.ndarray | None = None
    t: int = field(init=False, default=0)
    batches: list[Batch] = field(init=False, default_factory=list)
    pipeline: list[tuple[int, float]] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        if self.lead_time < 1:
            raise ValueError("lead_time must be at least 1 day")

    @property
    def horizon(self) -> int:
        """Number of days in the episode."""
        return len(self.demand)

    @property
    def on_hand(self) -> float:
        """Units physically in stock."""
        return float(sum(b.qty for b in self.batches))

    @property
    def on_order(self) -> float:
        """Units ordered but not yet received."""
        return float(sum(q for _, q in self.pipeline))

    def reset(self) -> dict:
        """Start a new episode and return the first observation."""
        self.t = 0
        self.pipeline = []
        self.batches = [Batch(self.params.shelf_life, float(self.initial_stock))] if self.initial_stock > 0 else []
        return self._obs()

    def _obs(self) -> dict:
        """What a policy may see before deciding today's order."""
        known = np.concatenate([self.history, self.demand[: self.t]])
        return {
            "t": self.t,
            "on_hand": self.on_hand,
            "on_order": self.on_order,
            "position": self.on_hand + self.on_order,
            "batches": [(b.expiry_day - self.t, b.qty) for b in self.batches],  # (days left, qty)
            "demand_history": known,
            "forecast": None if self.forecasts is None else self.forecasts[self.t],
            "params": self.params,
            "lead_time": self.lead_time,
        }

    def step(self, order_qty: float) -> tuple[dict | None, float, bool, dict]:
        """Advance one day.

        Returns:
            ``(next_obs, reward, done, info)``. ``reward`` is minus the day's
            cost (lost margin + expired stock at cost + holding cost).
        """
        t, p = self.t, self.params
        order_qty = max(0.0, float(order_qty))
        if order_qty > 0:
            self.pipeline.append((t + self.lead_time, order_qty))

        received = sum(q for d, q in self.pipeline if d == t)
        self.pipeline = [(d, q) for d, q in self.pipeline if d != t]
        if received > 0:
            self.batches.append(Batch(t + p.shelf_life, received))

        expired = sum(b.qty for b in self.batches if b.expiry_day <= t)
        self.batches = [b for b in self.batches if b.expiry_day > t]
        self.batches.sort(key=lambda b: b.expiry_day)

        demand = float(self.demand[t])
        remaining = demand
        for b in self.batches:  # FEFO
            take = min(b.qty, remaining)
            b.qty -= take
            remaining -= take
            if remaining <= 0:
                break
        self.batches = [b for b in self.batches if b.qty > 1e-9]
        sold = demand - remaining
        lost = remaining

        end_stock = self.on_hand
        holding = end_stock * p.holding_cost_per_day
        cost = lost * p.margin + expired * p.unit_cost + holding
        info = {
            "t": t, "order": order_qty, "received": received, "demand": demand, "sold": sold,
            "lost": lost, "expired": expired, "on_hand": end_stock, "on_order": self.on_order,
            "inventory_value": end_stock * p.unit_cost, "holding_cost": holding,
            "lost_value": lost * p.price, "lost_margin": lost * p.margin,
            "expired_value": expired * p.unit_cost, "cost": cost,
        }
        self.t += 1
        done = self.t >= self.horizon
        return (None if done else self._obs()), -cost, done, info


Policy = Callable[[dict], float]


@dataclass
class AverageBasedPolicy:
    """How many stores order today: top up to ``average daily sales x coverage days``.

    The average is the last 28 days of sales. Orders are placed every
    ``review_period`` days. It ignores trends, weekly patterns and expiry.
    """

    coverage_days: int
    review_period: int = 7

    def __call__(self, obs: dict) -> float:
        if obs["t"] % self.review_period:
            return 0.0
        avg = float(np.mean(obs["demand_history"][-28:]))
        target = avg * self.coverage_days
        return float(max(0, math.ceil(target - obs["position"])))


@dataclass
class ForecastBasedPolicy:
    """Order up to forecast demand over lead time + review period, plus safety stock.

    Safety stock comes from the forecast's own uncertainty: the P10-P90 band
    gives a daily spread, combined over the protection window and scaled to
    the chosen service level. When ``expiry_aware`` is on, the order-up-to
    level is capped at what is expected to sell before a new batch expires.
    """

    service_level: float
    review_period: int = 7
    expiry_aware: bool = True

    def __call__(self, obs: dict) -> float:
        if obs["t"] % self.review_period:
            return 0.0
        fc = obs["forecast"]
        if fc is None:
            raise ValueError("ForecastBasedPolicy needs forecasts in the environment")
        window = obs["lead_time"] + self.review_period
        H = fc.shape[0]
        mean = _extend(fc[:, FC_MEAN], window)
        sigma_day = _extend((fc[:, FC_P90] - fc[:, FC_P10]) / (2 * Z_80), window)
        z = _norm_ppf(self.service_level)
        target = mean.sum() + z * math.sqrt(float((sigma_day**2).sum()))
        if self.expiry_aware:
            sell_window = obs["lead_time"] + obs["params"].shelf_life
            sellable = _extend(fc[:, FC_MEAN], sell_window).sum() if sell_window > H else fc[:sell_window, FC_MEAN].sum()
            target = min(target, sellable)
        return float(max(0, math.ceil(target - obs["position"])))


def _extend(x: np.ndarray, n: int) -> np.ndarray:
    """First ``n`` values of ``x``, padding beyond its end with its mean."""
    if n <= len(x):
        return x[:n]
    return np.concatenate([x, np.full(n - len(x), x.mean())])


def _norm_ppf(p: float) -> float:
    """Standard normal quantile (Acklam's rational approximation, |error| < 1e-8 after refinement)."""
    if not 0 < p < 1:
        raise ValueError("service level must be in (0, 1)")
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02, 1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02, 6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00, -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    lo = 0.02425
    if p < lo:
        q = math.sqrt(-2 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    elif p > 1 - lo:
        q = math.sqrt(-2 * math.log(1 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    else:
        q = p - 0.5
        r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    e = 0.5 * math.erfc(-x / math.sqrt(2)) - p  # one Newton step
    return x - e * math.sqrt(2 * math.pi) * math.exp(x * x / 2)


@dataclass
class SimulationInputs:
    """Everything needed to (re)run the simulation, precomputed once."""

    items: list[str]
    demand: np.ndarray  # [n_items, T]
    history: np.ndarray  # [n_items, n_hist]
    forecasts: np.ndarray  # [n_items, T, H, 4]
    prices: np.ndarray  # [n_items] average selling price over the period
    dates: np.ndarray  # [T]


def run_episode(env: InventoryEnv, policy: Policy) -> pd.DataFrame:
    """Run one item through a full episode and return the daily log."""
    obs = env.reset()
    rows = []
    done = False
    while not done:
        obs, _, done, info = env.step(policy(obs))
        rows.append(info)
    return pd.DataFrame(rows)


def make_policies(cfg: SimConfig) -> dict[str, Policy]:
    """The two ordering rules compared in the project, keyed by display name."""
    return {
        "Average based": AverageBasedPolicy(cfg.lead_time + cfg.review_period + cfg.safety_days, cfg.review_period),
        "Forecast based": ForecastBasedPolicy(cfg.service_level, cfg.review_period, cfg.expiry_aware),
    }


def simulate(inputs: SimulationInputs, cfg: SimConfig, policies: dict[str, Policy] | None = None) -> pd.DataFrame:
    """Simulate every item under every policy; returns the daily log for all of them."""
    params = assign_item_params(inputs.items, inputs.prices, cfg)
    policies = policies or make_policies(cfg)
    logs = []
    for i, p in enumerate(params):
        # Same starting stock for every policy: enough for lead time + one review period.
        start = math.ceil(inputs.history[i, -28:].mean() * (cfg.lead_time + cfg.review_period))
        for name, policy in policies.items():
            env = InventoryEnv(inputs.demand[i], inputs.history[i], p, cfg.lead_time, start, inputs.forecasts[i])
            log_df = run_episode(env, policy)
            log_df["item_id"] = p.item_id
            log_df["policy"] = name
            log_df["shelf_life"] = p.shelf_life
            logs.append(log_df)
    out = pd.concat(logs, ignore_index=True)
    out["date"] = inputs.dates[out["t"].to_numpy()]
    return out


def summarize(log_df: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Business KPIs per policy (and optionally per item)."""
    keys = ["policy"] + (by or [])
    g = log_df.groupby(keys, sort=False)
    n_days = log_df["t"].nunique()
    out = pd.DataFrame({
        "stockout_days": g["lost"].apply(lambda s: int((s > 1e-9).sum())),
        "lost_units": g["lost"].sum(),
        "lost_sales_value": g["lost_value"].sum(),
        "lost_margin": g["lost_margin"].sum(),
        "expired_units": g["expired"].sum(),
        "expired_value": g["expired_value"].sum(),
        # Average over days of the total inventory value (summed across items).
        "avg_inventory_value": g["inventory_value"].sum() / n_days,
        "holding_cost": g["holding_cost"].sum(),
        "units_sold": g["sold"].sum(),
        "units_demanded": g["demand"].sum(),
        "units_ordered": g["order"].sum(),
    })
    out["fill_rate"] = out["units_sold"] / out["units_demanded"]
    out["total_cost"] = out["lost_margin"] + out["expired_value"] + out["holding_cost"]
    return out.reset_index()
