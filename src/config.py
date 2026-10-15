"""Central configuration for the Forecast to Order project.

Everything that a user might want to change (store, number of items, horizon,
simulation economics) lives here so that the pipeline, tests and dashboard all
read the same values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DataConfig:
    """Where the raw M5 files live and which slice of them to use."""

    data_dir: Path = ROOT / "data"
    processed_dir: Path = ROOT / "data" / "processed"
    # The evaluation file (d_1..d_1941) is preferred when present; the
    # validation file (d_1..d_1913) is used as a fallback.
    sales_files: tuple[str, ...] = ("sales_train_evaluation.csv", "sales_train_validation.csv")
    calendar_file: str = "calendar.csv"
    prices_file: str = "sell_prices.csv"
    store_id: str = "CA_1"
    n_items: int = 50

    @property
    def state_id(self) -> str:
        """State code derived from the store id (e.g. CA_1 -> CA)."""
        return self.store_id.split("_")[0]


@dataclass(frozen=True)
class ModelConfig:
    """Forecasting and backtest settings."""

    horizon: int = 28
    n_folds: int = 4  # rolling origins, each 28 days apart, ending at the last day
    first_train_origin: int = 120  # first origin (day index) used for training rows
    train_origin_stride: int = 5  # stride between training origins (coprime with 7)
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    seed: int = 42
    lgb_params: dict = field(
        default_factory=lambda: {
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_data_in_leaf": 100,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l2": 1.0,
            "verbose": -1,
        }
    )
    num_boost_round: int = 300


@dataclass(frozen=True)
class SimConfig:
    """Economics and operating rules of the inventory simulation."""

    lead_time: int = 7  # days between placing and receiving an order
    review_period: int = 7  # orders are placed once a week
    shelf_life_choices: tuple[int, ...] = (30, 60, 90, 180)
    shelf_life_multiplier: float = 1.0
    cost_ratio: float = 0.70  # unit cost as a share of the selling price
    annual_holding_rate: float = 0.25  # capital + storage cost per year, share of unit cost
    service_level: float = 0.98  # typical target for pharmacy / pharma distribution
    safety_days: int = 7  # extra days of cover in the "average based" rule
    expiry_aware: bool = True  # forecast policy caps orders by what can sell before expiry
    seed: int = 42


@dataclass(frozen=True)
class Config:
    """Top-level configuration bundle."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    outputs_dir: Path = ROOT / "outputs"


DEFAULT_CONFIG = Config()
