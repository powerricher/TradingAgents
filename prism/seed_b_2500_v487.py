"""PRISM v4.87: 2500-name VCP signal expansion and frozen forward diagnostics.
Discovery only; NOT an executable Seed A/B portfolio or independent OOS.
"""
import json,html
from pathlib import Path
import numpy as np,pandas as pd,yfinance as yf
from .seed_b_discovery_v442 import discover
from .seed_b_v443 import bch_confirm
from .atr_frontier_v433 import CUTOFF
END="2026-10-07"
def main():
 out=Path("results/prism-v487");out.mkdir(parents=True,exist_ok=True)
 names=pd.read_csv("results/prism-v479/universe_snapshot.csv").ticker.astype(str).tolist()
 if len(names)!=2500:raise RuntimeError("Expected frozen 2500 ticker snapshot")
 q=yf.download("QQQ",start="2021-01-01",end=END,auto_adjust=True,progress=False)
 if isinstance(q.columns,pd.MultiIndex):q.columns=q.columns.get_level_values(0)
 if q.empty:raise RuntimeError("QQQ unavailable")
 q.index=pd.DatetimeIndex(q.index).tz_localize(None).normalize()
 rows=[];errors=[];loaded=0
 for start in range(0,len(names),25):
  batch=names[start:start+25]
  try:raw=yf.download(batch,start="2021-01-01",end=END,auto_adjust=True,group_by="ticker",threads=False,progress=False)
  except Exception as e:
   errors.extend({"ticker":t,"error":str(e)} for t in batch);continue
  for t in batch:
   try:
    d=(raw[t] if isinstance(raw.columns,pd.MultiIndex) else raw).dropna(subset=["Open","High","Low","Close","Volume"]).copy()
    d.index=pd.DatetimeIndex(d.index).tz_localize(None).normalize()
    if len(d)<250:raise ValueError("Insufficient history")
    loaded+=1
    events=[(x["date"],x["edge"],"VCP_BREAKOUT") for x in discover(d,q.Close,t) if x["detector"]=="VCP_BREAKOUT"]
    events += [(day,edge,"BCH_CONFIRMED") for day,edge in bch_confirm(d)]
    for day,edge,detector in events:
     i=d.index.get_loc(day)
     if i+61>=len(d):continue
     ep=float(d.Open.iloc[i+1])
     if ep<=0 or not np.isfinite(ep):continue
     v=d.Volume.astype(float);prior=v.iloc[i-20:i].mean()
     close=d.Close.astype(float)
     ma50=close.iloc[i-49:i+1].mean()
     ma200=close.iloc[i-199:i+1].mean()
     for h in (5,20,60):
      pass
     rows.append({"ticker":t,"detector":detector,"signal_date":str(day.date()),
       "entry_date":str(d.index[i+1].date()),"edge":float(edge),
       "signal_rvol":float(v.iloc[i]/prior) if prior>0 else np.nan,
       "close_over_ma50":float(close.iloc[i]/ma50-1),
       "close_over_ma200":float(close.iloc[i]/ma200-1),
       "prior_20d_return":float(close.iloc[i]/close.iloc[i-20]-1),
       "entry_price":ep,
       **{f"forward_{h}d":float(close.iloc[i+1+h]/ep-1) for h in (5,20,60)}})
   except Exception as e:errors.append({"ticker":t,"error":str(e)})
  print("v4.87 processed",min(start+25,len(names)),"of",len(names),flush=True)
 df=pd.DataFrame(rows)
 (out/"coverage_errors.json").write_text(json.dumps(errors,indent=2))
 if df.empty:raise RuntimeError("No candidate signals")
 if df.duplicated(["ticker","detector","signal_date"]).any():raise RuntimeError("Duplicate signals")
 df.to_csv(out/"signal_features.csv",index=False)
 stats=[]
 for (det,year),g in df.groupby(["detector",pd.to_datetime(df.signal_date).dt.year]):
  for h in (5,20,60):
   r=g[f"forward_{h}d"].dropna()
   lo,hi=r.quantile([.05,.95])
   stats.append({"detector":det,"year":int(year),"horizon":h,"signals":len(r),
    "mean":float(r.mean()),"median":float(r.median()),"win_rate":float((r>0).mean()),
    "winsor_mean":float(r.clip(lo,hi).mean())})
 pd.DataFrame(stats).to_csv(out/"yearly_signal_quality.csv",index=False)
 report={"version":"v4.87","status":"2500_NAME_SEED_B_DISCOVERY_NOT_PORTFOLIO_NOT_OOS",
 "requested":len(names),"loaded":loaded,"coverage_errors":len(errors),
 "signals":len(df),"unique_tickers":int(df.ticker.nunique()),
 "detector_counts":df.detector.value_counts().to_dict(),
 "limitations":["2500-name snapshot is current-listed, not historical point-in-time; survivorship bias.",
 "Seed B VCP/BCH frozen detectors only; Seed A universe not expanded in this step.",
 "Forward 5/20/60-day close outcomes are independent signals, not executable exits.",
 "Only signals with full 60-day forward horizon retained, creating endpoint exclusion.",
 "Pre-signal features recorded separately from future returns; no predictive weights fitted.",
 "No shared cash, short sale, intraday fills, or independent OOS.",
 "Annual 200% is a target, not a forecast."]}
 (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
 (out/"report.html").write_text("<!doctype html><html lang='ko'><meta charset='utf-8'><title>PRISM v4.87</title><style>body{background:#0b1729;color:#eaf4ff;font:15px system-ui;margin:30px}h1{color:#5de4bd}</style><h1>PRISM v4.87 · Seed B 2500 Discovery</h1><pre>"+html.escape(json.dumps(report,ensure_ascii=False,indent=2))+"</pre></html>",encoding="utf-8")
 print(json.dumps({"status":"V487_COMPLETE","signals":len(df),"loaded":loaded}))
if __name__=="__main__":main()
