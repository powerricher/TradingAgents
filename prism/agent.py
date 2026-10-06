"""PRISM scoring agent built on TradingAgents' completed research state."""

from __future__ import annotations

import json
import re
from pydantic import BaseModel, Field

from tradingagents.agents.structured import bind_structured, invoke_structured
from .models import PrismScoreComponents


class PrismAssessment(BaseModel):
    catalyst_type: str = "unknown"
    catalyst_summary: str = ""
    catalyst_quality: float = Field(ge=0, le=100)
    event_probability: float = Field(ge=0, le=100)
    success_probability: float = Field(ge=0, le=100)
    fundamental_impact: float = Field(ge=0, le=100)
    surprise_edge: float = Field(ge=0, le=100)
    priced_in_edge: float = Field(ge=0, le=100)
    price_elasticity: float = Field(ge=0, le=100)
    positioning: float = Field(ge=0, le=100)
    market_regime: float = Field(ge=0, le=100)
    liquidity_risk: float = Field(ge=0, le=100)
    evidence_notes: str = ""

    def components(self) -> PrismScoreComponents:
        names = PrismScoreComponents.__dataclass_fields__
        return PrismScoreComponents(**{name: getattr(self, name) for name in names})


def _fallback_json(text: str) -> PrismAssessment:
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        raise ValueError("PRISM assessor returned no JSON object")
    return PrismAssessment.model_validate(json.loads(match.group(0)))


def create_prism_assessor(llm):
    structured = bind_structured(llm, PrismAssessment, "PRISM Assessor")

    def node(state) -> PrismAssessment:
        evidence = {
            "market_report": state.get("market_report", ""),
            "sentiment_report": state.get("sentiment_report", ""),
            "news_report": state.get("news_report", ""),
            "fundamentals_report": state.get("fundamentals_report", ""),
            "investment_plan": state.get("investment_plan", ""),
            "trader_plan": state.get("trader_investment_plan", ""),
            "final_decision": state.get("final_trade_decision", ""),
        }
        prompt = f"""You are the PRISM evidence assessor. Evaluate ONLY information present in the supplied reports.
Analysis date: {state.get('trade_date')}
Ticker: {state.get('company_of_interest')}

Score each field from 0 to 100. Do not calculate the final PRISM score.
A high liquidity_risk score means GOOD liquidity / LOW execution risk, so higher is better.
For event_probability, score the probability that the identified catalyst occurs as described.
For success_probability, score conditional success if it occurs.
For priced_in_edge, high means the opportunity appears underpriced/not fully priced in.
For surprise_edge, high means favorable asymmetry versus prevailing expectations.
If evidence is missing, use 50 rather than inventing facts.

Evidence:
{json.dumps(evidence, ensure_ascii=False)}

Return the PrismAssessment schema only."""
        result = invoke_structured(structured, prompt, "PRISM Assessor")
        if result is not None:
            return result
        return _fallback_json(llm.invoke(prompt + "\nReturn one JSON object and nothing else.").content)

    return node
