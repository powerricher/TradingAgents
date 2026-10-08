"""v4.5 technical-only 50-stock challenger. Entry model remains D+1 baseline."""
import argparse
from .universe_v45 import UNIVERSE
from .rolling_v42 import run
def main():
 p=argparse.ArgumentParser();p.add_argument("--output",default="results/prism-v45/report.html");a=p.parse_args()
 run(UNIVERSE,.10,a.output,.35)
if __name__=="__main__":main()
