"""CLI entry point: python -m prism TICKER YYYY-MM-DD."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from tradingagents.default_config import DEFAULT_CONFIG

from .runner import run_prism_m1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run PRISM M1 on a ticker and historical date.")
    parser.add_argument("ticker", help="Ticker, e.g. NVDA")
    parser.add_argument("date", help="Analysis date in YYYY-MM-DD")
    parser.add_argument("--output", default="prism_result.json", help="Result JSON path")
    args = parser.parse_args(argv)

    if not os.getenv("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY is required. Store it as a GitHub Actions secret for cloud runs.")

    config = dict(DEFAULT_CONFIG)
    config["llm_provider"] = "openai"
    # Use the repository's current curated defaults unless explicitly overridden.
    if os.getenv("PRISM_QUICK_MODEL"):
        config["quick_think_llm"] = os.environ["PRISM_QUICK_MODEL"]
    if os.getenv("PRISM_DEEP_MODEL"):
        config["deep_think_llm"] = os.environ["PRISM_DEEP_MODEL"]

    result = run_prism_m1(args.ticker.upper(), args.date, config=config)
    prediction = result["prism_prediction"]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # State contains LangChain objects and is intentionally excluded from the portable result.
    portable = {
        "ticker": args.ticker.upper(),
        "analysis_date": args.date,
        "tradingagents_signal": result["tradingagents_signal"],
        "prism_prediction": prediction,
    }
    output.write_text(json.dumps(portable, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 60)
    print(f"PRISM M1 | {args.ticker.upper()} | {args.date}")
    print("-" * 60)
    print(f"TradingAgents signal : {result['tradingagents_signal']}")
    print(f"PRISM score          : {prediction['prism_score']:.2f}")
    print(f"PRISM decision       : {prediction['decision']}")
    print(f"Catalyst             : {prediction['catalyst_type']}")
    print(f"Capital / trade      : KRW {prediction['capital_krw']:,}")
    print(f"Exit rule            : {prediction['exit_rule']}")
    print(f"Frozen prediction    : {prediction['prediction_id']}")
    print(f"Portable result      : {output}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
