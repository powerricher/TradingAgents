from __future__ import annotations
import argparse
from .audit_v42 import run
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--source",default="results/prism-v42/report.json")
 p.add_argument("--output",default="results/prism-v42-audit/report.html")
 a=p.parse_args(); run(a.source,a.output)
if __name__=="__main__": main()
