"""PRISM v4.56 — Falling Wedge structural reversal entry challengers.
All decisions at a completed close; entry at next session open.
Historical research only, no independent OOS.
"""
import argparse,json,html
from pathlib import Path
import numpy as np,pandas as pd
from .trend_v48 import load_prices,portfolio
from .trend_v49 import exit_trade
from .atr_frontier_v433 import CUTOFF
from .universe_v445 import UNIVERSE
START=10_000_000.
MODES=("IMMEDIATE","HIGHER_LOW","RETEST","RS_RECOVERY","STRUCTURAL_COMBINED")
def verify(d,q,signal,mode):
 i=d.index.get_loc(pd.Timestamp(signal))
 if mode=="IMMEDIATE":return i
 c=d.Close.astype(float);h=d.High.astype(float);l=d.Low.astype(float)
 rs=c.pct_change(10)-q.pct_change(10).reindex(d.index)
 # All criteria evaluated using only completed bars up to j.
 for j in range(i+1,min(i+11,len(d)-1)):
  if j<35:continue
  pre_low=float(l.iloc[max(0,i-10):i+1].min())
  recent_low=float(l.iloc[max(i+1,j-3):j+1].min())
  higher_low=recent_low>=pre_low*1.02 and float(c.iloc[j])>float(c.iloc[i])
  # Signal-day close is a conservative proxy for unknown wedge resistance;
  # this is a price retest, NOT exact fitted trendline retest.
  anchor=float(c.iloc[i])
  retest=(float(l.iloc[i+1:j+1].min())<=anchor*1.02 and
   float(c.iloc[j])>=anchor and float(c.iloc[j])>float(c.iloc[j-1]))
  rs_recovery=(pd.notna(rs.iloc[j]) and pd.notna(rs.iloc[i]) and
   float(rs.iloc[j])>float(rs.iloc[i])+.02 and float(c.iloc[j])>anchor)
  checks={"HIGHER_LOW":higher_low,"RETEST":retest,"RS_RECOVERY":rs_recovery,
   "STRUCTURAL_COMBINED":sum((higher_low,retest,rs_recovery))>=2}
  if checks[mode]:return j
 return None
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--candidates",default="results/prism-v452/candidate_exits.csv")
 p.add_argument("--seed-a",default="results/prism-v436/daily_equity.csv")
 p.add_argument("--out",default="results/prism-v456")
 a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 raw=pd.read_csv(a.candidates)
 wedge=raw[raw.pattern=="FALLING_WEDGE"].drop_duplicates(["ticker","signal_date"]).copy()
 if wedge.empty:raise RuntimeError("No frozen wedge signals")
 prices={};errors=[]
 for ticker in wedge.ticker.unique():
  try:prices[ticker]=load_prices(ticker,"2021-01-01").loc[lambda x:x.index<=CUTOFF]
  except Exception as exc:errors.append({"ticker":ticker,"error":str(exc)})
 if errors:raise RuntimeError("Frozen candidate price histories missing: "+str(errors))
 q=load_prices("QQQ","2021-01-01").loc[lambda x:x.index<=CUTOFF].Close
 seed=pd.read_csv(a.seed_a)
 seed=seed[seed.variant=="ATR4_CONTROL"].copy()
 seed.date=pd.to_datetime(seed.date).dt.normalize()
 aa=seed.set_index("date").equity.sort_index()
 if aa.empty or aa.index.duplicated().any():raise RuntimeError("Seed A unavailable")
 candidates=[];rejected=[];path=[]
 for r in wedge.itertuples(index=False):
  d=prices[r.ticker]
  if pd.Timestamp(r.signal_date) not in d.index:raise RuntimeError("Missing signal "+r.ticker)
  for mode in MODES:
   j=verify(d,q,r.signal_date,mode)
   if j is None:
    rejected.append({"ticker":r.ticker,"signal_date":r.signal_date,"mode":mode,
     "original_gross_return":float(r.exit_price/r.entry_price-1)})
    continue
   if mode=="IMMEDIATE":
    entry=r.entry_date;xd=r.exit_date;px=float(r.exit_price);ep=float(r.entry_price);reason=r.reason
   else:
    try:z=exit_trade(d,d.index[j],"CONFIRM30_1")
    except (ValueError,KeyError):z=None
    if z is None:
     rejected.append({"ticker":r.ticker,"signal_date":r.signal_date,"mode":mode,
      "original_gross_return":float(r.exit_price/r.entry_price-1),"reason":"UNRESOLVED_EXIT"})
     continue
    e,x,ep,px,reason=z;entry=str(d.index[e].date());xd=str(d.index[x].date())
   candidates.append({"ticker":r.ticker,"mode":mode,"signal_date":r.signal_date,
    "entry_date":entry,"exit_date":xd,"entry_price":float(ep),"exit_price":float(px),
    "edge":float(r.edge),"reason":reason,"delay_sessions":int(j-d.index.get_loc(pd.Timestamp(r.signal_date)))})
   path.append({"ticker":r.ticker,"mode":mode,"signal_date":r.signal_date,
    "entry_date":entry,"original_entry_price":float(r.entry_price),
    "confirmed_entry_price":float(ep),"original_gross_return":float(r.exit_price/r.entry_price-1)})
 c=pd.DataFrame(candidates)
 if c.empty:raise RuntimeError("No confirmed entries")
 c.to_csv(out/"confirmed_candidates.csv",index=False)
 pd.DataFrame(rejected).to_csv(out/"rejected_candidates.csv",index=False)
 pd.DataFrame(path).to_csv(out/"entry_delay_audit.csv",index=False)
 report={"version":"v4.56","status":"HISTORICAL STRUCTURAL REVERSAL CHALLENGERS",
  "original_signals":len(wedge),"strategies":{},
  "limitations":[
   "Same historical Falling Wedge signals, Seed A ATR4 frozen.",
   "Higher Low uses recent lows versus pre-signal lows; not full swing-pivot confirmation.",
   "Retest uses signal-day close as resistance proxy, not reconstructed wedge trendline.",
   "Relative strength is 10-session stock minus QQQ return; +2pp recovery hypothesis.",
   "Confirmation can occur within 10 trading days; buys next session open, never at confirmation close.",
   "Skipped profitable candidates reported; post-confirmation exit is recomputed.",
   "Not independent OOS; retrospectively selected universe and repeatedly researched data.",
   "Daily close MDD omits intraday drawdown; stop fills are hypothetical."]}
 ledger_all=[];curve_all=[]
 for mode in MODES:
  z=c[c["mode"]==mode]
  if z.empty:
   report["strategies"][mode]={"status":"NO_SIGNALS"};continue
  rows=z[["ticker","entry_date","exit_date","entry_price","exit_price","edge","reason"]].to_dict("records")
  perf,ledger,curve=portfolio(rows,prices,mode,rotation=False)
  bb=pd.DataFrame(curve).assign(date=lambda x:pd.to_datetime(x.date).dt.normalize()).set_index("date").equity
  calendar=aa.index.union(bb.index).sort_values()
  xa=aa.reindex(calendar);xb=bb.reindex(calendar)
  xa.loc[calendar<aa.index.min()]=START;xb.loc[calendar<bb.index.min()]=START
  if xa.isna().any() or xb.isna().any():raise RuntimeError("Internal equity gaps "+mode)
  total=xa+xb;peak=np.maximum.accumulate(np.r_[2*START,total.to_numpy()])[1:]
  missing=[x for x in rejected if x["mode"]==mode]
  report["strategies"][mode]={"seed_b":perf,"combined_return":float(total.iloc[-1]/(2*START)-1),
   "combined_mdd":float(np.min(total.to_numpy()/peak-1)),
   "eligible":len(z),"rejected":len(missing),
   "rejected_original_positive":sum(x["original_gross_return"]>0 for x in missing)}
  for x in ledger:x["mode"]=mode
  for x in curve:x["mode"]=mode
  ledger_all+=ledger;curve_all+=curve
 pd.DataFrame(ledger_all).to_csv(out/"ledger.csv",index=False)
 pd.DataFrame(curve_all).to_csv(out/"daily_equity.csv",index=False)
 (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
 (out/"report.html").write_text("<!doctype html><meta charset='utf-8'><title>PRISM v4.56</title><style>body{background:#0b1729;color:#eaf4ff;font:15px system-ui;margin:28px}h1{color:#5de4bd}pre{white-space:pre-wrap}</style><h1>PRISM v4.56 · Structural Reversal Confirmation</h1><pre>"+html.escape(json.dumps(report,ensure_ascii=False,indent=2))+"</pre>",encoding="utf-8")
 print(json.dumps({"status":"generated","strategies":list(report["strategies"])}))
if __name__=="__main__":main()
