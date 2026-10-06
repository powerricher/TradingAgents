"""Typed data contracts for PRISM M1 frozen predictions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


Decision = Literal["BUY", "PASS"]


@dataclass(frozen=True)
class PrismScoreComponents:
    """Normalized component scores on a 0-100 scale.

    Agents may propose these values with evidence, but the final weighted score
    is calculated deterministically by PrismScoringEngine.
    """

    catalyst_quality: float
    event_probability: float
    success_probability: float
    fundamental_impact: float
    surprise_edge: float
    priced_in_edge: float
    price_elasticity: float
    positioning: float
    market_regime: float
    liquidity_risk: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class FrozenPrediction:
    """Immutable pre-outcome prediction record.

    Outcome data intentionally does not belong here. It is stored separately so
    no future price or event result can mutate the original prediction.
    """

    prediction_id: str
    ticker: str
    analysis_date: str
    catalyst_type: str
    catalyst_summary: str
    components: PrismScoreComponents
    prism_score: float
    decision: Decision
    entry_rule: str = "analysis_date_close"
    exit_rule: str = "next_trading_day_close"
    capital_krw: int = 10_000_000
    frozen: bool = True

    def as_dict(self) -> dict:
        data = asdict(self)
        data["schema_version"] = "prism-m1-v1"
        return data
