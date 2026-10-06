"""Programmatic PRISM M1 runner using TradingAgents as the research engine."""

from __future__ import annotations

from pathlib import Path

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

from .config import TRADINGAGENTS_OVERRIDES
from .pipeline import PrismM1Pipeline


def run_prism_m1(
    ticker: str,
    analysis_date: str,
    *,
    config: dict | None = None,
    selected_analysts=("market", "social", "news", "fundamentals"),
):
    """Run TradingAgents, calculate deterministic PRISM score, then freeze it."""
    cfg = dict(config or DEFAULT_CONFIG)
    cfg.update(TRADINGAGENTS_OVERRIDES)

    graph = TradingAgentsGraph(selected_analysts, config=cfg)
    state, tradingagents_signal = graph.propagate(ticker, analysis_date)

    frozen_dir = Path(cfg["results_dir"]) / "prism" / "frozen"
    prism = PrismM1Pipeline(graph.deep_thinking_llm, frozen_dir)
    prediction = prism.freeze_from_state(state)

    return {
        "tradingagents_signal": tradingagents_signal,
        "prism_prediction": prediction.as_dict(),
        "state": state,
    }
