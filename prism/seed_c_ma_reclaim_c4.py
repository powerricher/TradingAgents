"""PRISM C4: sequential MA20/50/120/200 reclaim research.
Signal at completed daily close; next-open entry. No future information in detection.
C4 is independent challenger, not merged into Seed A/B/C production.
"""
import json,html
from pathlib import Path
import numpy as np,pandas as pd,yfinance as yf
from .seed_c_execution_v468 import exit_path
MA=(20,50,120,200)
def detect(d,max_span=60):
 c=d.Close.astype(float)
 ma={n:c.rolling(n,min_periods=n).mean() for n in MA}
 crosses={n:(c>ma[n])&(c.shift(1)<=ma[n].shift(1)) for n in MA}
 stage=0;start=None;last=None;events=[]
 for i in range(201,len(d)-6):
  if stage and i-start>max_span:stage=0;start=None
  crossed=[n for n in MA if bool(crosses[n].iloc[i])]
  if stage==0:
   if crossed==[20]:stage=1;start=i
  elif stage<4:
   expected=MA[stage]
   if expected in crossed:
    if len(crossed)!=1:
     stage=0;start=None;continue
    stage+=1
   elif crossed and any(MA.index(n)>stage for n in crossed):
    stage=0;start=None
   if stage and c.iloc[i]<ma[20].iloc[i]:
    stage=0;start=None
  if stage==4:
   if i+6<len(d):
    events.append({"signal_date":str(d.index[i].date()),
      "entry_date":str(d.index[i+1].date()),"entry_price":float(d.Open.iloc[i+1]),
      "span":i-start,"signal_close":float(c.iloc[i])})
   stage=0;start=None
 return events
def main():
 out=Path("results/prism-c4-ma-reclaim");out.mkdir(parents=True,exist_ok=True)
 names=pd.read_csv("results/prism-v467/universe_snapshot.csv").ticker.astype(str).tolist()
 rows=[];failures=[]
 for j in range(0,len(names),25):
  batch=names[j:j+25]
  try:allprices=yf.download(batch,start="2021-01-01",end="2026-10-07",auto_adjust=True,progress=False,threads=False,group_by="ticker")
  except Exception as exc:
   failures.extend({"ticker":t,"error":str(exc)} for t in batch);continue
  for t in batch:
   try:
    d=allprices[t] if isinstance(allprices.columns,pd.MultiIndex) else allprices
    d=d.dropna(subset=["Open","High","Low","Close","Volume"]).copy()
    d.index=pd.DatetimeIndex(d.index).tz_localize(None).normalize()
    if len(d)<220:continue
    for event in detect(d):
     ep=event["entry_price"]
     if not np.isfinite(ep) or ep<=0:continue
     xd,xp,reason=exit_path(d,pd.Timestamp(event["entry_date"]),ep,.08,.04,5)
     rows.append({"ticker":t,**event,"exit_date":xd,"exit_price":float(xp),
      "exit_reason":reason,"gross_return":float(xp/ep-1)})
   except Exception as exc:failures.append({"ticker":t,"error":str(exc)})
  print("C4 scanned",min(j+25,len(names)),"/",len(names),flush=True)
 df=pd.DataFrame(rows,columns=["ticker","signal_date","entry_date","entry_price","span","signal_close","exit_date","exit_price","exit_reason","gross_return"])
 df.to_csv(out/"signals.csv",index=False)
 (out/"errors.json").write_text(json.dumps(failures,indent=2))
 report={"status":"C4_EXPLORATORY_NOT_OOS","requested":len(names),"signals":len(df),
  "unique_tickers":int(df.ticker.nunique()) if len(df) else 0,
  "mean_gross_return":float(df.gross_return.mean()) if len(df) else None,
  "win_rate":float((df.gross_return>0).mean()) if len(df) else None,
  "limitations":["Strict sequential single-cross events; simultaneous MA crosses excluded.",
  "MA20 must remain supported through progression; reset on invalidation.",
  "Current-listed universe is survivorship-biased; no point-in-time constituents.",
  "Candidate independent returns, not a portfolio backtest.",
  "Daily OHLC assumes stop-first on ambiguous bars; no minute-data confirmation.",
  "No live orders or automatic strategy adoption."]}
 (out/"summary.json").write_text(json.dumps(report,indent=2))
 (out/"report.html").write_text("<!doctype html><html lang='ko'><meta charset='utf-8'><title>PRISM C4</title><style>body{background:#0d192b;color:#e8f3ff;font:15px system-ui;margin:30px}h1{color:#6ee7b7}</style><h1>PRISM C4 · Sequential MA Reclaim</h1><pre>"+html.escape(json.dumps(report,indent=2))+"</pre></html>")
 print(json.dumps(report))
if __name__=="__main__":main()
