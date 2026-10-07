from __future__ import annotations
import argparse
from .fast_lab import backtest_ticker
from .fast_report import write_report

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("tickers",help="Comma-separated tickers, e.g. NVDA,AAPL")
    p.add_argument("--start",required=True);p.add_argument("--end",required=True)
    p.add_argument("--threshold",type=float,default=70);p.add_argument("--output",default="results/prism-fast/report.html")
    a=p.parse_args(argv)
    signals=[]
    for t in [x.strip().upper() for x in a.tickers.split(",") if x.strip()]:
        signals.extend(backtest_ticker(t,a.start,a.end,a.threshold))
    signals.sort(key=lambda x:(x.date,x.ticker))
    s=write_report(signals,a.output)
    print(f"PRISM LAB FAST: trades={s['trades']} win_rate={s['win_rate']} avg_return={s['avg_return']} net_pnl_krw={s['net_pnl_krw']}")
    return 0
