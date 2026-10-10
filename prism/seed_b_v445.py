"""PRISM v4.45: Seed B controlled 80/100/150 candidate discovery.
Frozen Seed A ATR4 TECH80. Same dates, entry/exit and rank logic for B.
"""
import argparse,json,html
from pathlib import Path
import numpy as np,pandas as pd
from .universe_v445 import UNIVERSE,CORE100
from .universe_v413 import UNIVERSE as CORE80
from .seed_b_discovery_v442 import discover
from .seed_b_v443 import bch_confirm,make_rows
from .trend_v48 import load_prices,portfolio
from .regime_adaptive_v436 import regimes
from .atr_frontier_v433 import CUTOFF
START=10_000_000.
def main():
 p=argparse.ArgumentParser();p.add_argument("--seed-a",default="results/prism-v436/daily_equity.csv");p.add_argument("--out",default="results/prism-v445");a=p.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 prices={};issues=[]
 for t in UNIVERSE:
  try:
   d=load_prices(t,"2021-01-01").loc[lambda x:x.index<=CUTOFF]
   if d.empty or d.index.min()>pd.Timestamp("2022-01-01"):raise ValueError("Insufficient historical coverage")
   prices[t]=d
  except Exception as e:issues.append({"ticker":t,"error":str(e)})
 if issues:
  (out/"coverage_errors.json").write_text(json.dumps(issues,indent=2))
  raise RuntimeError("Incomplete 150-stock coverage: "+str(issues[:10]))
 q=load_prices("QQQ","2021-01-01").loc[lambda x:x.index<=CUTOFF].Close
 market=regimes(pd.Timestamp("2022-01-01"))
 allsignals=[]
 for t,d in prices.items():
  for x in discover(d,q,t):
   if x["detector"]=="VCP_BREAKOUT":allsignals.append(x)
  for day,edge in bch_confirm(d):
   allsignals.append({"ticker":t,"date":day,"detector":"BCH_CONFIRMED","edge":edge})
 signals=pd.DataFrame(allsignals)
 if signals.empty:raise RuntimeError("No signals")
 signals["entry_regime"]=signals.date.map(market)
 if signals.entry_regime.isna().any():raise RuntimeError("Unknown market regime")
 rows=make_rows(signals.to_dict("records"),prices)
 candidates=pd.DataFrame(rows)
 candidates["detector"]=signals.detector.to_numpy() if len(candidates)==len(signals) else ""
 if (candidates.detector=="").any():raise RuntimeError("Candidate identity mismatch")
 candidates["entry_regime"]=signals.entry_regime.to_numpy()
 seed=pd.read_csv(a.seed_a)
 seed=seed[seed.variant=="ATR4_CONTROL"].copy()
 seed.date=pd.to_datetime(seed.date).dt.normalize()
 aa=seed.set_index("date").equity.sort_index()
 if aa.empty or aa.index.duplicated().any():raise RuntimeError("Missing Seed A")
 groups={"TECH80":set(CORE80),"TECH100":set(CORE100),"TECH150":set(UNIVERSE)}
 report={"version":"v4.45","status":"RETROSPECTIVE / NOT INDEPENDENT OOS",
 "universe_counts":{k:len(v) for k,v in groups.items()},"results":{},
 "limitations":[
 "Same fixed Seed A TECH80 ATR4 account across all comparisons; only Seed B universe changes.",
 "All Seed B variants use same signal rules and MA30 exit, no retraining.",
 "Universe is retrospectively chosen; ticker delistings and historical point-in-time membership not represented.",
 "Some later-listed tickers have shorter histories, though pre-2022 coverage is required.",
 "Signal cutoff excludes last 220 calendar days; no forward validation.",
 "Cross-detector priority is a predeclared heuristic, not calibrated probability.",
 "Legacy portfolio engine terminal valuation is synthetic; Seed A may include unrealized gain.",
 "Daily-close MDD excludes intraday risk."]}
 curves=[];ledgers=[];audit=[]
 for label,allowed in groups.items():
  for mode in ("VCP_BREAKOUT","BCH_CONFIRMED","BCH_FIRST","BREAKOUT_FIRST","REGIME_ADAPTIVE"):
   z=candidates[candidates.ticker.isin(allowed)].copy()
   if mode in ("VCP_BREAKOUT","BCH_CONFIRMED"):z=z[z.detector==mode]
   if z.empty:raise RuntimeError("No signals "+label+mode)
   def score(r):
    if mode=="BCH_FIRST":return (100 if r.detector=="BCH_CONFIRMED" else 0)+r.edge
    if mode=="BREAKOUT_FIRST":return (100 if r.detector=="VCP_BREAKOUT" else 0)+r.edge
    if mode=="REGIME_ADAPTIVE":
     preferred="VCP_BREAKOUT" if r.entry_regime=="BULL" else "BCH_CONFIRMED"
     return (100 if r.detector==preferred else 0)+r.edge
    return r.edge
   records=[]
   for r in z.itertuples(index=False):
    records.append({"ticker":r.ticker,"entry_date":r.entry_date,"exit_date":r.exit_date,
     "entry_price":float(r.entry_price),"exit_price":float(r.exit_price),"edge":float(score(r)),"reason":r.reason})
   perf,ledger,curve=portfolio(records,prices,label+"_"+mode,rotation=False)
   bb=pd.DataFrame(curve).assign(date=lambda x:pd.to_datetime(x.date).dt.normalize()).set_index("date").equity
   calendar=aa.index.union(bb.index).sort_values()
   xa=aa.reindex(calendar);xb=bb.reindex(calendar)
   xa.loc[calendar<aa.index.min()]=START;xb.loc[calendar<bb.index.min()]=START
   if xa.isna().any() or xb.isna().any():raise RuntimeError("Internal calendar gaps")
   total=xa+xb;peak=np.maximum.accumulate(np.r_[2*START,total.to_numpy()])[1:]
   key=label+"_"+mode
   report["results"][key]={"seed_b":perf,"combined_return":float(total.iloc[-1]/(2*START)-1),
    "combined_mdd":float(np.min(total.to_numpy()/peak-1)),
    "signals":len(z),"executed_tickers":len(set(r["ticker"] for r in ledger))}
   for x in ledger:x["experiment"]=key
   ledgers+=ledger
   audit.extend({"date":str(day.date()),"experiment":key,"seed_a_equity":float(av),"seed_b_equity":float(bv),
    "total_equity":float(av+bv)} for day,av,bv in zip(calendar,xa,xb))
 signals.to_csv(out/"discovered_signals.csv",index=False)
 candidates.to_csv(out/"candidate_exits.csv",index=False)
 pd.DataFrame(ledgers).to_csv(out/"ledger.csv",index=False)
 pd.DataFrame(audit).to_csv(out/"daily_equity.csv",index=False)
 (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
 trs="".join(f"<tr><td>{html.escape(k)}</td><td>{v['seed_b']['return']:.2%}</td><td>{v['combined_return']:.2%}</td><td>{v['combined_mdd']:.2%}</td></tr>" for k,v in report["results"].items())
 (out/"report.html").write_text("<!doctype html><meta charset='utf-8'><title>PRISM v4.45</title><style>body{background:#0b1729;color:#eaf4ff;font:15px system-ui;margin:28px}h1{color:#5de4bd}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #345}</style><h1>PRISM v4.45 · Seed B Universe 150</h1><p>Frozen Seed A · Same Seed B rules · Historical, not OOS.</p><table><tr><th>Experiment</th><th>Seed B</th><th>Combined</th><th>MDD</th></tr>"+trs+"</table>",encoding="utf-8")
 print(json.dumps({"status":"generated","experiments":len(report["results"])}))
if __name__=="__main__":main()
