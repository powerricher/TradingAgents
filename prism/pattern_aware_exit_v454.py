"""PRISM v4.54 — frozen long-pattern entries, staged exit and giveback audit.
No hindsight signals. Seed A unchanged. Historical research, not OOS.
"""
import argparse,json,html
from pathlib import Path
import numpy as np,pandas as pd
from .trend_v48 import load_prices,portfolio,STOP_SLIPPAGE
from .trend_v47 import atr
from .atr_frontier_v433 import CUTOFF
from .universe_v445 import UNIVERSE,CORE100
from .universe_v413 import UNIVERSE as CORE80
START=10_000_000.
MODES=("MA30_CONTROL","ATR4_CONTROL","EARLY_PROTECTION","TWO_STAGE_ATR","PROFIT_GIVEBACK")
def staged_exit(d,entry,mode):
 e=d.index.get_loc(pd.Timestamp(entry))
 entry_price=float(d.Open.iloc[e])
 if not np.isfinite(entry_price) or entry_price<=0:raise ValueError("Invalid entry")
 a=atr(d);peak=entry_price;stop=-np.inf;activated=False
 for j in range(e,len(d)):
  op=float(d.Open.iloc[j]);lo=float(d.Low.iloc[j]);cl=float(d.Close.iloc[j])
  days=j-e
  if j>e and np.isfinite(stop):
   if op<=stop:return str(d.index[j].date()),op,"gap_stop"
   if lo<=stop:return str(d.index[j].date()),stop*(1-STOP_SLIPPAGE),"stop_touch_stress"
  if days>=180 and j+1<len(d):
   return str(d.index[j+1].date()),float(d.Open.iloc[j+1]),"max_hold"
  # Intraday high is not used to update a stop that executes the same day.
  peak=max(peak,cl)
  if mode=="EARLY_PROTECTION":
   k=2.0 if days<10 else 3.0
  elif mode=="TWO_STAGE_ATR":
   if peak>=entry_price*1.08:activated=True
   k=4.0 if activated else 2.0
  else:
   k=4.0
  if pd.notna(a.iloc[j]):
   stop=max(stop,peak-k*float(a.iloc[j]))
  if mode=="PROFIT_GIVEBACK" and peak>=entry_price*1.10:
   stop=max(stop,entry_price+(peak-entry_price)*.50)
 return str(d.index[-1].date()),float(d.Close.iloc[-1]),"terminal_mark_to_market"
def early_path(d,entry):
 e=d.index.get_loc(pd.Timestamp(entry));p=float(d.Open.iloc[e])
 z={"entry_date":str(d.index[e].date())}
 for n in (3,5,10,20,60,120):
  if e+n>=len(d):
   z.update({f"ret_{n}":None,f"mae_{n}":None,f"mfe_{n}":None});continue
  s=d.iloc[e:e+n+1]
  z.update({f"ret_{n}":float(d.Close.iloc[e+n]/p-1),
   f"mae_{n}":float(s.Low.min()/p-1),f"mfe_{n}":float(s.High.max()/p-1)})
 return z
def main():
 p=argparse.ArgumentParser();p.add_argument("--source",default="results/prism-v453/exit_candidate_comparison.csv")
 p.add_argument("--seed-a",default="results/prism-v436/daily_equity.csv")
 p.add_argument("--out",default="results/prism-v454");args=p.parse_args()
 out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
 source=pd.read_csv(args.source)
 source=source[source["mode"]=="MA30_ORIGINAL"].copy()
 source=source.drop_duplicates(["ticker","entry_date","pattern"])
 if source.empty:raise RuntimeError("No frozen entries")
 prices={};errors=[]
 for ticker in source.ticker.unique():
  try:prices[ticker]=load_prices(ticker,"2021-01-01").loc[lambda x:x.index<=CUTOFF]
  except Exception as exc:errors.append({"ticker":ticker,"error":str(exc)})
 if errors:raise RuntimeError("Missing OHLC for frozen candidates: "+str(errors))
 baseline=pd.read_csv(args.source)
 audit=[];rows=[];unresolved=[]
 for r in source.itertuples(index=False):
  d=prices[r.ticker]
  if pd.Timestamp(r.entry_date) not in d.index:raise RuntimeError("Entry missing "+r.ticker)
  audit.append({"ticker":r.ticker,"pattern":r.pattern,**early_path(d,r.entry_date)})
  for mode in MODES:
   if mode=="MA30_CONTROL":z=baseline[(baseline.ticker==r.ticker)&(baseline.entry_date==r.entry_date)&(baseline["mode"]=="MA30_ORIGINAL")&(baseline.pattern==r.pattern)]
   elif mode=="ATR4_CONTROL":z=baseline[(baseline.ticker==r.ticker)&(baseline.entry_date==r.entry_date)&(baseline["mode"]=="ATR4")&(baseline.pattern==r.pattern)]
   else:z=None
   if mode in ("MA30_CONTROL","ATR4_CONTROL"):
    if z.empty:
     unresolved.append({"ticker":r.ticker,"entry_date":r.entry_date,"mode":mode});continue
    x=z.iloc[0];xd=str(x.exit_date);px=float(x.exit_price);reason=str(x.reason)
   else:xd,px,reason=staged_exit(d,r.entry_date,mode)
   rows.append({"ticker":r.ticker,"pattern":r.pattern,"entry_date":str(r.entry_date),
    "exit_date":xd,"entry_price":float(r.entry_price),"exit_price":px,
    "edge":float(r.edge),"reason":reason,"mode":mode,
    "gross_return":float(px/float(r.entry_price)-1)})
 pd.DataFrame(audit).to_csv(out/"early_failure_paths.csv",index=False)
 pd.DataFrame(rows).to_csv(out/"exit_candidates.csv",index=False)
 pd.DataFrame(unresolved,columns=["ticker","entry_date","mode"]).to_csv(out/"unresolved.csv",index=False)
 seed=pd.read_csv(args.seed_a);seed=seed[seed.variant=="ATR4_CONTROL"].copy()
 seed.date=pd.to_datetime(seed.date).dt.normalize()
 aa=seed.set_index("date").equity.sort_index()
 if aa.empty or aa.index.duplicated().any():raise RuntimeError("Seed A unavailable")
 df=pd.DataFrame(rows)
 groups={"TECH80":set(CORE80),"TECH100":set(CORE100),"TECH150":set(UNIVERSE)}
 report={"version":"v4.54","status":"HISTORICAL EXIT RESEARCH / NOT INDEPENDENT OOS",
 "frozen_entries":len(source),"unresolved":len(unresolved),"strategies":{},
 "limitations":[
  "Same v4.53 entry candidates across exit variants; no entry filter changes.",
  "Two-stage activation uses closing prices only; trailing stops execute next session or later.",
  "Early protection uses ATR2 first 10 sessions; then ATR3. No same-day high-based fills.",
  "Profit giveback activates after +10% closing gain and protects half of close-based peak profit.",
  "Terminal marks at last available close are unrealized hypothetical liquidation.",
  "No commission-free or guaranteed stop fills; stop-touch uses 20bps stress haircut.",
  "Retrospective TECH80/100/150 universe and repeatedly researched sample; not OOS.",
  "One position per Seed, no rotation; opportunity costs need further counterfactual audit.",
  "Daily-close MDD excludes intraday lows."]}
 ledgers=[];curves=[]
 for group,allowed in groups.items():
  for pattern in ("ASCENDING_TRIANGLE","DESCENDING_TRIANGLE_REVERSAL","FALLING_WEDGE","ALL_LONG_PATTERNS"):
   for mode in MODES:
    z=df[(df.ticker.isin(allowed))&(df["mode"]==mode)]
    if pattern!="ALL_LONG_PATTERNS":z=z[z.pattern==pattern]
    name=f"{group}_{pattern}_{mode}"
    if z.empty:
     report["strategies"][name]={"status":"NO_SIGNALS"};continue
    records=z[["ticker","entry_date","exit_date","entry_price","exit_price","edge","reason"]].to_dict("records")
    try:perf,ledger,curve=portfolio(records,prices,name,rotation=False)
    except Exception as exc:
     report["strategies"][name]={"status":"INVALID","error":str(exc)};continue
    bb=pd.DataFrame(curve).assign(date=lambda x:pd.to_datetime(x.date).dt.normalize()).set_index("date").equity
    cal=aa.index.union(bb.index).sort_values()
    xa=aa.reindex(cal);xb=bb.reindex(cal)
    xa.loc[cal<aa.index.min()]=START;xb.loc[cal<bb.index.min()]=START
    if xa.isna().any() or xb.isna().any():raise RuntimeError("Internal valuation gap")
    total=xa+xb;peak=np.maximum.accumulate(np.r_[2*START,total.to_numpy()])[1:]
    report["strategies"][name]={"seed_b":perf,"combined_return":float(total.iloc[-1]/(2*START)-1),
     "combined_mdd":float(np.min(total.to_numpy()/peak-1)),
     "candidates":len(records)}
    for x in ledger:x["strategy"]=name
    for x in curve:x["strategy"]=name
    ledgers+=ledger;curves+=curve
 pd.DataFrame(ledgers).to_csv(out/"ledger.csv",index=False)
 pd.DataFrame(curves).to_csv(out/"daily_equity.csv",index=False)
 (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
 (out/"report.html").write_text("<!doctype html><meta charset='utf-8'><title>PRISM v4.54</title><style>body{background:#0b1729;color:#eaf4ff;font:15px system-ui;margin:28px}h1{color:#5de4bd}pre{white-space:pre-wrap}</style><h1>PRISM v4.54 · Staged Exit Research</h1><pre>"+html.escape(json.dumps(report,ensure_ascii=False,indent=2))+"</pre>",encoding="utf-8")
 print(json.dumps({"status":"generated","strategies":len(report["strategies"]),"entries":len(source)}))
if __name__=="__main__":main()
