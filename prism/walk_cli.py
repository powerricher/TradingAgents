from __future__ import annotations
import argparse
from .calibrate import run

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("tickers")
    p.add_argument("--train-start",default="2026-01-01")
    p.add_argument("--train-end",default="2026-06-30")
    p.add_argument("--val-start",default="2026-07-01")
    p.add_argument("--val-end",default="2026-10-06")
    p.add_argument("--threshold",type=float,default=50.0)
    p.add_argument("--output",default="results/prism-walk/report.html")
    a=p.parse_args(argv)
    tickers=[x.strip().upper() for x in a.tickers.split(",") if x.strip()]
    m=run(tickers,a.train_start,a.train_end,a.val_start,a.val_end,a.threshold,a.output)
    print(m)
if __name__=="__main__": raise SystemExit(main())
