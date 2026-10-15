"""PRISM v4.70 — frozen C1 market-regime and position-sizing audit.
No changes to entry/exit signals. Not independent out-of-sample.
"""
import json,html
from pathlib import Path
import numpy as np,pandas as pd
from .regime_adaptive_v436 import regimes
from .trend_v48 import COST
START=10_000_000.
NAME="C1_VOL_EXPANSION_TP8_SL4_H5"
POLICIES={
 "FULL_CONTROL":lambda r:1.,
 "RISK_1PCT":lambda r:.25,
 "RISK_2PCT":lambda r:.50,
 "BEAR_CASH":lambda r:0. if r=="BEAR" else 1.,
 "REGIME_RISK":lambda r:0. if r=="BEAR" else (.25 if r=="SIDEWAYS" else .50)
}
def simulate(rows,dates,policy):
 candidates={}
 for r in rows:
  candidates.setdefault(r["entry_date"],[]).append(r)
 for k in candidates:
  candidates[k].sort(key=lambda x:(-float(x["edge"]),x["ticker"]))
 cash=START;pos=None;ledger=[];curve=[];skipped=0
 for day in dates:
  key=day.strftime("%Y-%m-%d");sold=False
  if pos is not None and pos["exit_date"]==key:
   cash+=pos["shares"]*pos["exit_price"]*(1-COST/2)
   ledger.append({"ticker":pos["ticker"],"entry_date":pos["entry_date"],
    "exit_date":key,"capital_after":cash,"reason":pos["reason"],
    "entry_regime":pos["entry_regime"],"allocation":pos["allocation"]})
   pos=None;sold=True
  available=candidates.get(key,[])
  if pos is None and not sold and available:
   best=available[0]
   fraction=float(policy(best["entry_regime"]))
   if fraction>0:
    budget=cash*fraction
    shares=budget*(1-COST/2)/best["entry_price"]
    cash-=budget
    pos={**best,"shares":shares,"allocation":fraction}
    if pos["exit_date"]==key:
     cash+=shares*pos["exit_price"]*(1-COST/2)
     ledger.append({"ticker":pos["ticker"],"entry_date":key,
      "exit_date":key,"capital_after":cash,"reason":pos["reason"],
      "entry_regime":pos["entry_regime"],"allocation":fraction})
     pos=None;sold=True
  executed_today=bool(pos is not None and pos["entry_date"]==key) or bool(sold and ledger and ledger[-1]["entry_date"]==key)
  skipped+=len(available)-int(executed_today)
  equity=cash
  if pos is not None:
   # Frozen daily mark-to-market from underlying OHLC is not available in
   # candidate CSV; obtain equivalent price path by loading historical OHLC.
   d=prices[pos["ticker"]]
   if day not in d.index:raise RuntimeError("Missing close "+pos["ticker"]+" "+key)
   equity+=pos["shares"]*float(d.loc[day,"Close"])
  curve.append({"date":key,"equity":float(equity)})
 if pos is not None:raise RuntimeError("Unclosed position")
 v=np.asarray([x["equity"] for x in curve]);peak=np.maximum.accumulate(np.r_[START,v])[1:]
 return {"return":float(cash/START-1),"final_krw":float(cash),
  "daily_close_mdd":float(np.min(v/peak-1)),"trades":len(ledger),
  "same_day_exits":sum(x["entry_date"]==x["exit_date"] for x in ledger),
  "skipped":skipped},ledger,curve
def main():
 global prices
 out=Path("results/prism-v470");out.mkdir(parents=True,exist_ok=True)
 candidates=pd.read_csv("results/prism-v468/candidate_exit_scenarios.csv")
 base=pd.read_csv("results/prism-v468/ledger.csv")
 original=pd.read_csv("results/prism-v468/daily_equity.csv")
 z=candidates[(candidates.detector=="C1_VOL_EXPANSION")&
  (candidates.exit_rule=="TP8_SL4")&(candidates.hold==5)].copy()
 l=base[base.strategy==NAME].copy()
 ref=original[original.strategy==NAME].copy()
 if z.empty or l.empty or ref.empty:raise RuntimeError("Missing v4.68 reference")
 from .trend_v48 import load_prices
 from .atr_frontier_v433 import CUTOFF
 prices={}
 for t in sorted(z.ticker.unique()):
  d=load_prices(t,"2021-01-01").loc[lambda x:x.index<=CUTOFF]
  prices[t]=d
 market=regimes(pd.Timestamp("2022-01-01"))
 z["entry_regime"]=pd.to_datetime(z.entry_date).map(market)
 if z.entry_regime.isna().any() or (z.entry_regime=="UNKNOWN").any():
  raise RuntimeError("Missing prior-close QQQ regime")
 z["edge"]=z["edge"].astype(float)
 records=z[["ticker","entry_date","exit_date","entry_price","exit_price","edge","reason","entry_regime"]].to_dict("records")
 dates=sorted(pd.Timestamp(x) for x in pd.to_datetime(ref.date).dt.normalize().unique())
 report={"version":"v4.70","status":"FROZEN_C1_RISK_AND_REGIME_RESEARCH_NOT_OOS",
  "candidate_regime_counts":z.entry_regime.value_counts().to_dict(),
  "strategies":{},"limitations":[
  "Entry and exit rules frozen at C1 TP8 SL4 H5; only allocation or entry-regime gate changes.",
  "Regime uses previous-close QQQ data from existing v4.36 regime logic.",
  "RISK_1PCT means 25% capital with a planned -4% stop; gap losses may exceed 1%.",
  "One position maximum, cash remains uninvested, no same-day reentry after exit.",
  "Market regime is entry-date label; not an exogenous independent market-state proof.",
  "Repeated 2022-2026 research and present-day universe selection; NOT independent OOS.",
  "Historical stop-first OHLC execution; no minute-by-minute order validation.",
  "Only daily-close MDD; no automatic trades or live alerts."]}
 ledgers=[];curves=[]
 for name,policy in POLICIES.items():
  perf,ledger,curve=simulate(records,dates,policy)
  report["strategies"][name]=perf
  for r in ledger:r["strategy"]=name
  for r in curve:r["strategy"]=name
  ledgers+=ledger;curves+=curve
 # Require full-capital baseline to reproduce the original engine before trusting sizing.
 control=report["strategies"]["FULL_CONTROL"]
 ref_last=float(ref.sort_values("date").equity.iloc[-1])
 if abs(control["final_krw"]-ref_last)>max(1.,ref_last*.00001):
  raise RuntimeError("Baseline replay mismatch: "+str(control["final_krw"]-ref_last))
 if control["trades"]!=len(l):
  raise RuntimeError("Baseline trade count mismatch")
 if len(report["strategies"])!=5:raise RuntimeError("Incomplete strategy audit")
 pd.DataFrame(ledgers).to_csv(out/"ledger.csv",index=False)
 pd.DataFrame(curves).to_csv(out/"daily_equity.csv",index=False)
 z.to_csv(out/"frozen_candidate_regimes.csv",index=False)
 (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
 trs="".join(f"<tr><td>{html.escape(k)}</td><td>{v['return']:.2%}</td><td>{v['daily_close_mdd']:.2%}</td><td>{v['trades']}</td></tr>" for k,v in report["strategies"].items())
 (out/"report.html").write_text("<!doctype html><html lang='ko'><meta charset='utf-8'><title>PRISM v4.70</title><style>body{background:#0b1729;color:#eaf4ff;font:15px system-ui;margin:28px}h1{color:#5de4bd}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #345}</style><h1>PRISM v4.70 · C1 Regime and Position Sizing</h1><p>Frozen signals; no live trading. Baseline must reconcile with v4.68.</p><table><tr><th>Policy</th><th>Return</th><th>Daily MDD</th><th>Trades</th></tr>"+trs+"</table></html>",encoding="utf-8")
 print(json.dumps({"status":"VALID","strategies":len(report["strategies"])}))
if __name__=="__main__":main()
