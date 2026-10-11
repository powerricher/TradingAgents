"""PRISM v4.55 — Frozen pattern robustness and causal delayed-entry research.
Historical research only; no fitted parameters or OOS claim.
"""
import argparse,json,html
from pathlib import Path
import numpy as np,pandas as pd
from .trend_v48 import load_prices,portfolio
from .trend_v49 import exit_trade
from .atr_frontier_v433 import CUTOFF
from .regime_adaptive_v436 import regimes
from .universe_v445 import UNIVERSE
START=10_000_000.
def confirm(d,signal,delay):
 i=d.index.get_loc(pd.Timestamp(signal))
 if i+delay+1>=len(d):return None
 # No lookahead: confirmation is known only after close of day i+delay.
 base=float(d.Close.iloc[i])
 segment=d.Close.iloc[i+1:i+delay+1]
 if len(segment)!=delay:return None
 if not (float(segment.iloc[-1])>base and float(segment.iloc[-1])>=float(segment.iloc[-2]) if delay>1 else float(segment.iloc[-1])>base):
  return None
 day=d.index[i+delay]
 z=exit_trade(d,day,"CONFIRM30_1")
 if z is None:return None
 e,x,entry,px,reason=z
 return {"entry_date":str(d.index[e].date()),"exit_date":str(d.index[x].date()),
 "entry_price":float(entry),"exit_price":float(px),"reason":reason}
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--candidates",default="results/prism-v452/candidate_exits.csv")
 p.add_argument("--exits",default="results/prism-v454/exit_candidates.csv")
 p.add_argument("--seed-a",default="results/prism-v436/daily_equity.csv")
 p.add_argument("--out",default="results/prism-v455")
 a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 original=pd.read_csv(a.candidates)
 exits=pd.read_csv(a.exits)
 for d in (original,exits):
  if d.empty:raise RuntimeError("Missing prior frozen results")
 prices={};errors=[]
 for t in original.ticker.unique():
  try:prices[t]=load_prices(t,"2021-01-01").loc[lambda x:x.index<=CUTOFF]
  except Exception as e:errors.append({"ticker":t,"error":str(e)})
 if errors:raise RuntimeError("Price coverage differs from frozen sample "+str(errors))
 seed=pd.read_csv(a.seed_a)
 seed=seed[seed.variant=="ATR4_CONTROL"].copy()
 seed.date=pd.to_datetime(seed.date).dt.normalize()
 aa=seed.set_index("date").equity.sort_index()
 if aa.empty:raise RuntimeError("Missing Seed A")
 market=regimes(pd.Timestamp("2022-01-01"))
 report={"version":"v4.55","status":"HISTORICAL RESEARCH / NOT INDEPENDENT OOS",
 "descending_stress":{},"falling_wedge":{},"opportunity_cost":{},
 "limitations":[
 "Winner removal is ex-post stress testing, not a tradable policy.",
 "Confirmation requires a positive close relative to original breakout and no decline versus prior close; fills next open.",
 "Nonconfirmed signals are omitted; their missed future winners must be counted separately.",
 "Delayed entries use a newly simulated MA30 exit, not the original exit date.",
 "Regime labels are entry-date labels, not realized P&L earned within each regime.",
 "No same-session reentry, no leverage; original one-position portfolio engine.",
 "Repeated 2022-2026 research, retrospective universe and missing delisted symbols mean not OOS.",
 "Daily-close MDD omits intraday troughs and modeled stop fills may differ from execution."]}
 ledgers=[];curves=[];candidate_rows=[]
 desc=exits[(exits.pattern=="DESCENDING_TRIANGLE_REVERSAL")&(exits["mode"]=="TWO_STAGE_ATR")]
 desc_records=desc[["ticker","entry_date","exit_date","entry_price","exit_price","edge","reason"]].to_dict("records")
 if not desc_records:raise RuntimeError("Missing descending Two-Stage records")
 perf,ledger,curve=portfolio(desc_records,prices,"DESC_TWO_STAGE",rotation=False)
 capital=START;attributions=[]
 for r in ledger:
  pnl=float(r["capital_after"])-capital;capital=float(r["capital_after"])
  attributions.append({"ticker":r["ticker"],"entry_date":r["entry_date"],"pnl":pnl,
   "entry_regime":str(market.get(pd.Timestamp(r["entry_date"]),"UNKNOWN"))})
 wins=sorted((r for r in attributions if r["pnl"]>0),key=lambda r:r["pnl"],reverse=True)
 report["descending_stress"]["baseline"]=perf
 report["descending_stress"]["entry_regime_pnl"]={str(k):float(sum(x["pnl"] for x in attributions if x["entry_regime"]==k)) for k in sorted({x["entry_regime"] for x in attributions})}
 report["descending_stress"]["top3_winners"]=wins[:3]
 for n in (1,2,3):
  excluded={(x["ticker"],x["entry_date"]) for x in wins[:n]}
  reduced=[r for r in desc_records if (r["ticker"],r["entry_date"]) not in excluded]
  res,_,_=portfolio(reduced,prices,"DESC_EX_TOP"+str(n),rotation=False)
  report["descending_stress"]["exclude_top"+str(n)]=res
 wedge=original[original.pattern=="FALLING_WEDGE"].copy()
 if wedge.empty:raise RuntimeError("No wedge signals")
 modes={"IMMEDIATE":[],"CONFIRM3":[],"CONFIRM5":[]}
 skipped={"CONFIRM3":[],"CONFIRM5":[]}
 for r in wedge.itertuples(index=False):
  basic={"ticker":r.ticker,"entry_date":r.entry_date,"exit_date":r.exit_date,
   "entry_price":float(r.entry_price),"exit_price":float(r.exit_price),
   "edge":float(r.edge),"reason":r.reason}
  modes["IMMEDIATE"].append(basic)
  for n in (3,5):
   z=confirm(prices[r.ticker],r.signal_date,n)
   if z is None:
    skipped["CONFIRM"+str(n)].append({"ticker":r.ticker,"signal_date":r.signal_date,
     "original_gross_return":float(r.exit_price/r.entry_price-1)})
   else:
    modes["CONFIRM"+str(n)].append({"ticker":r.ticker,"edge":float(r.edge),**z})
 for mode,records in modes.items():
  if not records:
   report["falling_wedge"][mode]={"status":"NO_TRADES"};continue
  result,lg,cv=portfolio(records,prices,"WEDGE_"+mode,rotation=False)
  report["falling_wedge"][mode]={"seed_b":result,"eligible":len(records)}
  for x in lg:x["strategy"]=mode
  for x in cv:x["strategy"]=mode
  ledgers+=lg;curves+=cv
  bb=pd.DataFrame(cv).assign(date=lambda x:pd.to_datetime(x.date).dt.normalize()).set_index("date").equity
  calendar=aa.index.union(bb.index).sort_values()
  xa=aa.reindex(calendar);xb=bb.reindex(calendar)
  xa.loc[calendar<aa.index.min()]=START;xb.loc[calendar<bb.index.min()]=START
  if xa.isna().any() or xb.isna().any():raise RuntimeError("Internal equity gap")
  total=xa+xb;peak=np.maximum.accumulate(np.r_[2*START,total.to_numpy()])[1:]
  report["falling_wedge"][mode]["combined_return"]=float(total.iloc[-1]/(2*START)-1)
  report["falling_wedge"][mode]["combined_mdd"]=float(np.min(total.to_numpy()/peak-1))
 for mode,items in skipped.items():
  report["falling_wedge"][mode]["rejected"]=len(items)
  report["falling_wedge"][mode]["rejected_original_positive"]=sum(x["original_gross_return"]>0 for x in items)
  candidate_rows.extend([{"mode":mode,**x} for x in items])
 # Opportunity blocking: audit other pattern entries during actual occupied intervals.
 opportunities=[]
 all_candidates=original[["ticker","entry_date","exit_date","pattern","entry_price","exit_price"]].drop_duplicates()
 for mode in modes:
  trades=[r for r in ledgers if r["strategy"]==mode]
  for r in trades:
   blocked=all_candidates[(all_candidates.entry_date>r["entry_date"])&(all_candidates.entry_date<r["exit_date"])]
   opportunities.append({"mode":mode,"held_ticker":r["ticker"],"entry_date":r["entry_date"],
    "exit_date":r["exit_date"],"blocked_signals":len(blocked),
    "blocked_positive_original_exit":int((blocked.exit_price>blocked.entry_price).sum())})
 report["opportunity_cost"]={mode:{"blocked_signals":sum(x["blocked_signals"] for x in opportunities if x["mode"]==mode),
  "blocked_positive_candidates":sum(x["blocked_positive_original_exit"] for x in opportunities if x["mode"]==mode)}
  for mode in modes}
 pd.DataFrame(ledgers).to_csv(out/"wedge_ledger.csv",index=False)
 pd.DataFrame(curves).to_csv(out/"wedge_daily_equity.csv",index=False)
 pd.DataFrame(attributions).to_csv(out/"descending_attribution.csv",index=False)
 pd.DataFrame(candidate_rows).to_csv(out/"rejected_wedge_candidates.csv",index=False)
 pd.DataFrame(opportunities).to_csv(out/"blocked_opportunities.csv",index=False)
 (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
 (out/"report.html").write_text("<!doctype html><meta charset='utf-8'><title>PRISM v4.55</title><style>body{background:#0b1729;color:#eaf4ff;font:15px system-ui;margin:28px}h1{color:#5de4bd}pre{white-space:pre-wrap}</style><h1>PRISM v4.55 · Pattern Robustness &amp; Opportunity Cost</h1><pre>"+html.escape(json.dumps(report,ensure_ascii=False,indent=2))+"</pre>",encoding="utf-8")
 print(json.dumps({"status":"generated","wedge_modes":list(report["falling_wedge"])}))
if __name__=="__main__":main()
