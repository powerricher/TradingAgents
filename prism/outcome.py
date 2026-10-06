"""D+1 outcome calculation for a frozen PRISM prediction."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from tradingagents.memory.settlement import fetch_returns, resolve_benchmark


@dataclass(frozen=True)
class PrismOutcome:
    ticker: str
    analysis_date: str
    holding_days: int
    raw_return: float
    alpha_return: float
    resolution_date: str
    capital_krw: int
    pnl_krw: int
    result: str

    def as_dict(self) -> dict:
        return asdict(self)


def resolve_d1_outcome(ticker: str, analysis_date: str, capital_krw: int, config: dict) -> PrismOutcome:
    benchmark = resolve_benchmark(ticker, config)
    raw, alpha, days, resolution_date = fetch_returns(
        ticker, analysis_date, holding_days=1, benchmark=benchmark
    )
    if raw is None:
        raise ValueError("D+1 outcome is not available yet")
    return PrismOutcome(
        ticker=ticker,
        analysis_date=analysis_date,
        holding_days=days,
        raw_return=raw,
        alpha_return=alpha,
        resolution_date=resolution_date,
        capital_krw=capital_krw,
        pnl_krw=round(capital_krw * raw),
        result="WIN" if raw > 0 else ("LOSS" if raw < 0 else "FLAT"),
    )
