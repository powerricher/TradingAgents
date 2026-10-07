"""Aggregation and HTML reporting for PRISM LAB FAST."""
from __future__ import annotations
import html, json
from collections import defaultdict
from pathlib import Path


def _stats(signals):
    n=len(signals); wins=sum(s.d1_return>0 for s in signals); rets=[s.d1_return for s in signals]
    return {"trades":n,"wins":wins,"losses":n-wins,
            "win_rate":wins/n if n else None,
            "avg_return":sum(rets)/n if n else None,
            "net_pnl_krw":sum(s.pnl_krw for s in signals)}


def summarize(signals):
    base=_stats(signals)
    buckets={}
    for lo,hi,label in [(70,75,"70-74"),(75,80,"75-79"),(80,85,"80-84"),(85,90,"85-89"),(90,101,"90+")]:
        z=[s for s in signals if lo<=s.score<hi]
        buckets[label]=_stats(z)
    by_ticker={}
    groups=defaultdict(list)
    for s in signals: groups[s.ticker].append(s)
    for ticker,z in sorted(groups.items()): by_ticker[ticker]=_stats(z)
    base["buckets"]=buckets
    base["by_ticker"]=by_ticker
    return base


def write_report(signals,path):
    s=summarize(signals)
    def pct(x): return "—" if x is None else f"{x:.2%}"
    bucket_rows="".join(f"<tr><td>{k}</td><td>{v['trades']}</td><td>{pct(v['win_rate'])}</td><td>{pct(v['avg_return'])}</td><td>₩{v['net_pnl_krw']:+,}</td></tr>" for k,v in s["buckets"].items())
    ticker_rows="".join(f"<tr><td>{html.escape(k)}</td><td>{v['trades']}</td><td>{v['wins']}</td><td>{v['losses']}</td><td>{pct(v['win_rate'])}</td><td>{pct(v['avg_return'])}</td><td>₩{v['net_pnl_krw']:+,}</td></tr>" for k,v in s["by_ticker"].items())
    trade_rows="".join(f"<tr><td>{html.escape(x.ticker)}</td><td>{x.date}</td><td>{x.score:.2f}</td><td>{x.entry_close:.2f}</td><td>{x.exit_close:.2f}</td><td>{pct(x.d1_return)}</td><td>{x.pnl_krw:+,}</td></tr>" for x in signals)
    doc=f"""<!doctype html><meta charset="utf-8"><title>PRISM LAB FAST v1 Baseline</title>
<style>body{{font-family:Arial,sans-serif;background:#0b1020;color:#e8eefc;margin:32px}}.cards{{display:flex;gap:12px;flex-wrap:wrap}}.card{{background:#151d35;padding:16px;border-radius:12px;min-width:150px}}table{{width:100%;border-collapse:collapse;margin-top:18px;background:#11182b}}th,td{{padding:10px;border-bottom:1px solid #27314d;text-align:right}}th:first-child,td:first-child{{text-align:left}}h1{{margin-bottom:4px}}small{{color:#9aabc8}}</style>
<h1>PRISM LAB FAST v1 — Baseline</h1><small>Zero-paid-LLM deterministic backtest · threshold ≥70 · D+1 close exit · ₩10M/independent trade</small>
<div class="cards"><div class="card">Trades<br><b>{s['trades']}</b></div><div class="card">Win rate<br><b>{pct(s['win_rate'])}</b></div><div class="card">Avg return<br><b>{pct(s['avg_return'])}</b></div><div class="card">Net P&L<br><b>₩{s['net_pnl_krw']:+,}</b></div></div>
<h2>By ticker</h2><table><tr><th>Ticker</th><th>N</th><th>Wins</th><th>Losses</th><th>Win rate</th><th>Avg D+1</th><th>Net P&L</th></tr>{ticker_rows}</table>
<h2>Score calibration</h2><table><tr><th>Score</th><th>N</th><th>Win rate</th><th>Avg D+1</th><th>Net P&L</th></tr>{bucket_rows}</table>
<h2>All trades</h2><table><tr><th>Ticker</th><th>Date</th><th>Score</th><th>Entry</th><th>Exit D+1</th><th>Return</th><th>P&L KRW</th></tr>{trade_rows}</table>"""
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(doc,encoding="utf-8")
    p.with_suffix(".json").write_text(json.dumps({"model":"PRISM LAB FAST v1","summary":s,"trades":[x.as_dict() for x in signals]},ensure_ascii=False,indent=2),encoding="utf-8")
    return s
