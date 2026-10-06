"""PRISM M1 execution constants."""

PRISM_MODEL_VERSION = "m1-v1"
BUY_THRESHOLD = 70.0
CAPITAL_PER_TRADE_KRW = 10_000_000
HOLDING_PERIOD_DAYS = 1

# TradingAgents settlement uses analysis-date close as entry and the Nth later
# trading-session close as exit. N=1 therefore implements PRISM's D+1 rule.
TRADINGAGENTS_OVERRIDES = {
    "holding_period_days": HOLDING_PERIOD_DAYS,
}
