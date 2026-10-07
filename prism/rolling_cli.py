import argparse
from .rolling_v41 import run
def main():
 p=argparse.ArgumentParser();p.add_argument("tickers");p.add_argument("--top-pct",type=float,default=.10);p.add_argument("--output",default="results/prism-v41/report.html");a=p.parse_args();ts=[x.strip().upper() for x in a.tickers.split(",") if x.strip()];print(run(ts,a.top_pct,a.output))
if __name__=="__main__":main()
