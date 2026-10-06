"""End-to-end PRISM M1 adapter around a completed TradingAgents run."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .agent import create_prism_assessor
from .models import FrozenPrediction
from .scoring import PrismScoringEngine
from .store import FrozenPredictionStore


class PrismM1Pipeline:
    def __init__(self, llm, frozen_dir: str | Path, scoring: PrismScoringEngine | None = None):
        self.assess = create_prism_assessor(llm)
        self.scoring = scoring or PrismScoringEngine()
        self.store = FrozenPredictionStore(frozen_dir)

    @staticmethod
    def prediction_id(ticker: str, date: str, catalyst_type: str, catalyst_summary: str) -> str:
        digest = hashlib.sha256(
            f"{ticker}|{date}|{catalyst_type}|{catalyst_summary}".encode("utf-8")
        ).hexdigest()[:12]
        return f"{ticker.upper()}_{date}_{digest}"

    def freeze_from_state(self, state: dict) -> FrozenPrediction:
        assessment = self.assess(state)
        components = assessment.components()
        score = self.scoring.score(components)
        prediction = FrozenPrediction(
            prediction_id=self.prediction_id(
                state["company_of_interest"], state["trade_date"],
                assessment.catalyst_type, assessment.catalyst_summary,
            ),
            ticker=state["company_of_interest"].upper(),
            analysis_date=state["trade_date"],
            catalyst_type=assessment.catalyst_type,
            catalyst_summary=assessment.catalyst_summary,
            components=components,
            prism_score=score,
            decision=self.scoring.decision(score),
        )
        self.store.freeze(prediction)
        return prediction
