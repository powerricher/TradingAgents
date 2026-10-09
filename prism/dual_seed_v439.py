"""PRISM v4.39: two independent 10M KRW strategy accounts.
Merge synchronized daily equity, no cross-funding, no leverage or rotation.
"""
import argparse,json,html
from pathlib import Path
import numpy as np,pandas as pd
SEED=10_000_000.
PAIRS={"MA30_ATR4":("MA30_CONTROL","ATR4_CONTROL"),
 "MA30_UNIFIED":("MA30_CONTROL","UNIFIED_FIXED"),
 "ATR4_UNIFIED":("ATR4_CONTROL","UNIFIED_FIXED")}
def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--control",default="results/prism-v436/daily_equity.csv")
 ap.add_argument("--unified",default="results/prism-v438/daily_equity.csv")
 ap.add_argument("--control-ledger",default="results/prism-v436/ledger.csv")
 ap.add_argument("--unified-ledger",default="results/prism-v438/ledger.csv")
 ap.add_argument("--out",default="results/prism-v439")
 a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 ctrl=pd.read_csv(a.control);unified=pd.read_csv(a.unified)
 ledctrl=pd.read_csv(a.control_ledger);leduni=pd.read_csv(a.unified_ledger)
 for x in (ctrl,unified):
  if not {"date","variant","equity"}.issubset(x.columns):raise RuntimeError("Missing daily equity fields")
  x.date=pd.to_datetime(x.date).dt.normalize()
 for x in (ledctrl,leduni):
  if "variant" not in x:raise RuntimeError("Missing ledger strategy")
 # All legs must be independently executable. Reuse already-tested frozen
 # strategy curves; never allocate 20M to each or recompute fills.
 legs={}
 for label,src in (("MA30_CONTROL",ctrl),("ATR4_CONTROL",ctrl),("UNIFIED_FIXED",unified)):
  z=src[src.variant==label][["date","equity"]].sort_values("date").copy()
  if z.empty or z.date.duplicated().any():raise RuntimeError("Missing or duplicate curve "+label)
  if not np.isfinite(z.equity).all():raise RuntimeError("Invalid equity "+label)
  legs[label]=z.set_index("date").equity
 report={"version":"v4.39","status":"HISTORICAL DUAL-SEED / NOT INDEPENDENT OOS",
  "initial_total_krw":2*SEED,"pairs":{},"notes":[
  "Two fully independent 10M KRW sleeves, no capital transfer, no leverage.",
  "Combined daily equity equals sum of the two independent daily marked equity curves.",
  "Same ticker may be held in both sleeves; this is not diversification guarantee.",
  "Original 80-stock universe only. 100-stock extension is not yet implemented.",
  "Unified Fixed has exact original CONFIRM30_1 bull exit, whereas legacy MA30 and ATR4 remain independent controls.",
  "Terminal mark-to-market may be synthetic liquidation in upstream engine.",
  "2022-2026 data repeatedly researched; no independent OOS."]}
 daily=[]
 for name,(a,b) in PAIRS.items():
  xa,xb=legs[a].align(legs[b],join="outer")
  # Calendar must match; don't silently fill missing valuation.
  if xa.isna().any() or xb.isna().any():raise RuntimeError("Mismatched trading days "+name)
  equity=xa+xb
  peak=np.maximum.accumulate(np.r_[2*SEED,equity.to_numpy()])[1:]
  mdd=float(np.min(equity.to_numpy()/peak-1))
  source_ledger=pd.concat([ledctrl,leduni],ignore_index=True)
  counts={leg:int((source_ledger.variant==leg).sum()) for leg in (a,b)}
  report["pairs"][name]={"seed_a":a,"seed_b":b,"final_krw":float(equity.iloc[-1]),
    "compound_return":float(equity.iloc[-1]/(2*SEED)-1),
    "daily_close_mdd":mdd,"seed_a_return":float(xa.iloc[-1]/SEED-1),
    "seed_b_return":float(xb.iloc[-1]/SEED-1),"trades":counts}
  daily.extend([{"date":str(day.date()),"pair":name,"seed_a_equity":float(va),
   "seed_b_equity":float(vb),"total_equity":float(va+vb)} for day,va,vb in zip(equity.index,xa,xb)])
 pd.DataFrame(daily).to_csv(out/"dual_daily_equity.csv",index=False)
 (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
 trs="".join(f"<tr><td>{html.escape(k)}</td><td>{v['compound_return']:.2%}</td><td>{v['daily_close_mdd']:.2%}</td><td>{v['final_krw']:,.0f}</td></tr>" for k,v in report["pairs"].items())
 (out/"report.html").write_text("<!doctype html><meta charset='utf-8'><title>PRISM v4.39</title><style>body{background:#0b1729;color:#eaf4ff;font:15px system-ui;margin:28px}h1{color:#5de4bd}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #345}</style><h1>PRISM v4.39 · Dual Seed Independent Portfolio</h1><p>2 independent 10M KRW sleeves. Historical research, not OOS.</p><table><tr><th>Pair</th><th>Return</th><th>Daily MDD</th><th>Final KRW</th></tr>"+trs+"</table>",encoding="utf-8")
 print(json.dumps({"status":"generated","pairs":report["pairs"]},ensure_ascii=False))
if __name__=="__main__":main()
