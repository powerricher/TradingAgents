# PRISM v4.19 — Price Action Intelligence
Research design, not investment advice or verified transcription of linked videos.

- **SMB Second Day:** deferred. Requires verified event timestamps, announcement timing and intraday prices; do not pretend an earnings gap is a verified event.
- **Qullamaggie breakout:** use 20/60-day prior highs, 20-day return, up-day persistence as *our own proxy definitions*.
- **Brian Shannon AVWAP:** event-anchored VWAP deferred. A separate rolling 20-day daily close-volume weighted **proxy** is tested and must never be labeled true AVWAP.
- **Minervini VCP:** 5/20-day return-volatility ratio, 10/20-day intraday range ratio, 5/20-day volume ratio as *our own proxies*, not official VCP detection.
- **Control:** chronological MA30-outcome Ridge with frozen TECH80 signal eligibility and identical MA30 exits.
- **No hindsight:** features available at signal close; enter next session open; training uses only exited historical trades with a five-day lag.
- **Ablation:** CONTROL_RIDGE / BREAKOUT / VCP / VWAP_PROXY / COMBINED.
- **Acceptance:** never promote based on a single historical compound-return maximum. Require new-date forward evaluation, calibration, concentration, slippage, and drawdown checks.
