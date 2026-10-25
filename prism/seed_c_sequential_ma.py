"""PRISM C4: sequential moving-average recovery, exploratory research.
Signals use only completed daily closes. No LOPE-specific fitting.
"""
import json,html
from pathlib import Path
import numpy as np,pandas as pd,yfinance as yf
from .seed_c_execution_v468 import exit_path
from .seed_c_multi_position_v471 import simulate
from .atr_frontier_v433 import CUTOFF
START=10_000_000.
WINDOWS=(20,40,60)
def detect(d,window,strict):
 c=d.Close.astype(float)
 ma={n:c.rolling(n,min_periods=n).mean() for n in (20,50,120,200)}
 cross={n:(c>ma[n])&(c.shift(1)<=ma[n].shift(1)) for n in ma}
 out=[]
 for i in range(201,len(d)-7):
  if not bool(cross[200].iloc[i]):continue
  if not all(c.iloc[i]>ma[n].iloc[i] for n in (20,50,120,200)):continue
  start=max(200,i-window+1)
  indices=[]
  for n in (20,50,120):
   eligible=np.flatnonzero(cross[n].iloc[start:i+1].fillna(False).to_numpy())
   if not len(eligible):break
   indices.append(start+int(eligible[-1]))
  if len(indices)!=3:continue
  if strict and not(indices[0]<indices[1]<indices[2]<i):continue
  if not strict and not(max(indices)<=i):continue
  out.append(i)
 return out
def main():
 out=Path("results/prism-c4-sequential-ma");out.mkdir(parents=True,exist_ok=True)
 names=pd.read_csv("results/prism-v467/universe_snapshot.csv").ticker.astype(str).tolist()
 if len(names)!=1000:raise RuntimeError("Missing frozen 1000-name universe")
 baseline=pd.read_csv("results/prism-v471/daily_equity.csv")
 baseline=baseline[baseline.strategy=="FOUR_25"].sort_values("date")
 if baseline.empty:raise RuntimeError("Missing v4.71 calendar")
 calendar=[pd.Timestamp(x) for x in pd.to_datetime(baseline.date).dt.normalize().unique()]
 prices={};errors=[];rows=[]
 for start in range(0,len(names),25):
  batch=names[start:start+25]
  try:data=yf.download(batch,start="2021-01-01",end="2026-10-07",auto_adjust=True,progress=False,threads=False,group_by="ticker")
  except Exception as exc:
   errors.extend({"ticker":t,"error":str(exc)} for t in batch);continue
  for t in batch:
   try:
    d=data[t] if isinstance(data.columns,pd.MultiIndex) else data
    d=d.dropna(subset=["Open","High","Low","Close","Volume"]).copy()
    d.index=pd.DatetimeIndex(d.index).tz_localize(None).normalize()
    if len(d)<210:continue
    prices[t]=d
    for window in WINDOWS:
     for strict in (True,False):
      mode=("STRICT" if strict else "FLEX")+"_"+str(window)
      for i in detect(d,window,strict):
       if d.index[i]<pd.Timestamp("2022-01-01") or i+6>=len(d):continue
       ep=float(d.Open.iloc[i+1])
       if ep<=0 or not np.isfinite(ep):continue
       z=exit_path(d,d.index[i+1],ep,.08,.04,5)
       if z is None:continue
       xd,xp,reason=z
       v=d.Volume.astype(float);rvol=float(v.iloc[i]/v.iloc[i-20:i].mean()) if v.iloc[i-20:i].mean()>0 else 0
       rows.append({"ticker":t,"mode":mode,"signal_date":str(d.index[i].date()),
        "entry_date":str(d.index[i+1].date()),"entry_price":ep,"exit_date":xd,
        "exit_price":xp,"reason":reason,"edge":rvol,"gross_return":xp/ep-1})
   except Exception as exc:errors.append({"ticker":t,"error":str(exc)})
  print("C4 processed",min(start+25,len(names)),flush=True)
 df=pd.DataFrame(rows)
 if df.empty:raise RuntimeError("No sequential MA signals")
 if df.duplicated(["ticker","mode","entry_date"]).any():raise RuntimeError("Duplicate candidate identity")
 df.to_csv(out/"candidate_exits.csv",index=False)
 (out/"coverage_errors.json").write_text(json.dumps(errors,indent=2))
 report={"status":"HISTORICAL EXPLORATORY NOT OOS","version":"C4","strategies":{},
  "limitations":["Current-listed 1000 names, not point-in-time constituents.",
   "Strict uses last observed cross for each MA in chronological order, which may omit earlier valid sequences.",
   "Flexible requires all three MA crosses within lookback and 200-day final cross.",
   "Close-confirmed signals, next open entry, fixed TP8 SL4 H5 stop-first daily OHLC fills.",
   "C4 is an independent challenger, not integrated with Seed A/B or C1.",
   "No optimization on LOPE and no independent out-of-sample.",
   "No automatic orders or alerts."]}
 ledgers=[];curves=[]
 for mode,z in df.groupby("mode"):
  records=z[["ticker","entry_date","exit_date","entry_price","exit_price","reason","edge"]].to_dict("records")
  perf,ledger,curve=simulate(records,calendar,prices,4,.25)
  report["strategies"][mode]={**perf,"candidate_count":len(z),
    "mean_candidate_gross":float(z.gross_return.mean())}
  for x in ledger:x["mode"]=mode
  for x in curve:x["mode"]=mode
  ledgers+=ledger;curves+=curve
 if len(report["strategies"])!=6:raise RuntimeError("Expected six MA variants")
 pd.DataFrame(ledgers).to_csv(out/"ledger.csv",index=False)
 pd.DataFrame(curves).to_csv(out/"daily_equity.csv",index=False)
 (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
 trs="".join(f"<tr><td>{html.escape(k)}</td><td>{v['candidate_count']}</td><td>{v['return']:.2%}</td><td>{v['mdd']:.2%}</td><td>{v['trades']}</td></tr>" for k,v in report["strategies"].items())
 (out/"report.html").write_text("<!doctype html><html lang='ko'><meta charset='utf-8'><title>PRISM C4</title><style>body{background:#0c1728;color:#edf5ff;font:15px system-ui;margin:28px}h1{color:#62e4ba}td,th{padding:10px;border-bottom:1px solid #345}table{border-collapse:collapse;width:100%}</style><h1>PRISM C4 · Sequential MA Breakout</h1><p>Strict and flexible 20/50/120/200 recovery, research only.</p><table><tr><th>Mode</th><th>Candidates</th><th>Return</th><th>MDD</th><th>Trades</th></tr>"+trs+"</table>",encoding="utf-8")
 print(json.dumps({"status":"VALID","variants":len(report["strategies"])}))
if __name__=="__main__":main()
