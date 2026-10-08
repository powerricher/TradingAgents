"""PRISM v4.3: 50-ticker candidate universe and frozen 2021-2026 rolling entries.
Experimental challenger; does not overwrite v4.2 champion.
"""
import argparse,json
from pathlib import Path
from .rolling_v42 import run as base_run

BASE="NVDA AAPL MSFT AMZN GOOGL META AVGO TSLA NFLX AMD ORCL PLTR CSCO IBM INTC QCOM TXN AMAT LRCX MU ARM CRWD PANW ADBE NOW SHOP UBER ABNB COIN HOOD".split()
ADDED="ANET VRT ALAB CRDO AXTI INDI MRVL CIEN APP NET DDOG SNOW VKTX CRNX IONS RARE MIRM SRRK PCVX IMTX".split()
UNIVERSE=BASE+ADDED
assert len(UNIVERSE)==50 and len(set(UNIVERSE))==50

def main():
 p=argparse.ArgumentParser();p.add_argument("--output",default="results/prism-v43/report.html");a=p.parse_args()
 meta=base_run(UNIVERSE,.10,a.output,.35)
 print("v4.3 universe:",len(UNIVERSE),"selected:",meta["pooled_selected"].get("n",0))
if __name__=="__main__":main()
