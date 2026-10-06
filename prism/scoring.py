"""Deterministic PRISM M1 score calculation."""

from __future__ import annotations

from .models import PrismScoreComponents

# Weights sum to 1.00. Changing them is a model version change and should be
# evaluated against frozen historical predictions rather than edited in place.
DEFAULT_WEIGHTS: dict[str, float] = {
    "catalyst_quality": 0.15,
    "event_probability": 0.10,
    "success_probability": 0.15,
    "fundamental_impact": 0.10,
    "surprise_edge": 0.10,
    "priced_in_edge": 0.15,
    "price_elasticity": 0.10,
    "positioning": 0.05,
    "market_regime": 0.05,
    "liquidity_risk": 0.05,
}

BUY_THRESHOLD = 70.0


class PrismScoringEngine:
    def __init__(self, weights: dict[str, float] | None = None, buy_threshold: float = BUY_THRESHOLD):
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        self.buy_threshold = float(buy_threshold)
        if abs(sum(self.weights.values()) - 1.0) > 1e-9:
            raise ValueError("PRISM weights must sum to 1.0")

    @staticmethod
    def _validate(value: float, name: str) -> float:
        value = float(value)
        if not 0.0 <= value <= 100.0:
            raise ValueError(f"{name} must be between 0 and 100, got {value}")
        return value

    def score(self, components: PrismScoreComponents) -> float:
        values = components.as_dict()
        missing = set(self.weights) - set(values)
        if missing:
            raise ValueError(f"Missing PRISM components: {sorted(missing)}")
        total = sum(self._validate(values[name], name) * weight for name, weight in self.weights.items())
        return round(total, 2)

    def decision(self, score: float) -> str:
        return "BUY" if float(score) >= self.buy_threshold else "PASS"
