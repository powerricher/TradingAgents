import argparse
from .risk_gate import run
def main():
 p=argparse.ArgumentParser();p.add_argument("tickers");p.add_argument("--train-start",default="2026-01-01");p.add_argument("--train-end",default="2026-06-30");p.add_argument("--val-start",default="2026-07-01");p.add_argument("--val-end",default="2026-10-06");p.add_argument("--max-down",type=float,default=.45);p.add_argument("--min-up",type=float,default=.50);p.add_argument("--output",default="results/prism-v4/report.html");a=p.parse_args();ts=[x.strip().upper() for x in a.tickers.split(",") if x.strip()];print(run(ts,a.train_start,a.train_end,a.val_start,a.val_end,a.max_down,a.min_up,a.output))
if __name__=="__main__":main()
