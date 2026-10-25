"""PRISM C4: sequential 20/50/120/200 SMA reclaim research.
Daily completed-bar signals, next-session open entry; historical and not OOS.
"""
import json,html
from pathlib import Path
import numpy as np,pandas as pd,yfinance as yf
from .seed_c_execution_v468 import exit_path
from .seed_c_multi_position_v471 import simulate
START="2021-01-01";END="2026-10-07"
def signals(d):
 c=d.Close.astype(float);v=d.Volume.astype(float)
 mas={n:c.rolling(n,min_periods=n).mean() for n in (20,50,120,200)}
 crosses={n:(c>mas[n])&(c.shift(1)<=mas[n].shift(1)) for n in mas}
 rv=v/v.shift(1).rolling(20).mean()
 # State machine: sequential crossings must be on distinct sessions, within 60 sessions.
 # Re-crossing is permitted; each new 20-cross restarts the sequence.
 state=0;first=None;dates=[];out=[]
 for i in range(200,len(d)-6):
  if state and i-first>60:state=0
  if bool(crosses[20].iloc[i]):
   state=1;first=i;dates=[i]
  elif state in (1,2,3):
   expected=(50,120,200)[state-1]
   if bool(crosses[expected].iloc[i]):
    dates.append(i);state+=1
    if state==4:
     if rv.iloc[i]>=1.5 and (c.iloc[max(0,first-120):first]<mas[200].iloc[max(0,first-120):first]).any():
      out.append((i,dates.copy(),float(rv.iloc[i])))
     state=0
 return out
def main():
 out=Path("results/prism-c4");out.mkdir(parents=True,exist_ok=True)
 names=pd.read_csv("results/prism-v467/universe_snapshot.csv").ticker.astype(str).tolist()
 if len(names)!=1000:raise RuntimeError("Frozen 1000-symbol universe unavailable")
 prices={};rows=[];errors=[]
 for k in range(0,len(names),25):
  batch=names[k:k+25]
  try:raw=yf.download(batch,start=START,end=END,auto_adjust=True,group_by="ticker",threads=False,progress=False)
  except Exception as exc:errors.extend({"ticker":t,"error":str(exc)} for t in batch);continue
  for t in batch:
   try:
    d=raw[t] if isinstance(raw.columns,pd.MultiIndex) else raw
    d=d.dropna(subset=["Open","High","Low","Close","Volume"]).copy()
    d.index=pd.DatetimeIndex(d.index).tz_localize(None).normalize()
    if len(d)<260:continue
    prices[t]=d
    for i,stages,rv in signals(d):
     if d.index[i]<pd.Timestamp("2022-01-01"):continue
     e=i+1;ep=float(d.Open.iloc[e])
     if ep<=0:continue
     x=exit_path(d,d.index[e],ep,.08,.04,5)
     if x is None:continue
     xd,xp,reason=x
     rows.append({"ticker":t,"signal_date":str(d.index[i].date()),
      "ma20_date":str(d.index[stages[0]].date()),"ma50_date":str(d.index[stages[1]].date()),
      "ma120_date":str(d.index[stages[2]].date()),"ma200_date":str(d.index[stages[3]].date()),
      "entry_date":str(d.index[e].date()),"entry_price":ep,
      "exit_date":xd,"exit_price":xp,"reason":reason,"edge":rv,
      "rvol":rv,"gross_return":xp/ep-1})
   except Exception as exc:errors.append({"ticker":t,"error":str(exc)})
  print("C4 processed",min(k+25,len(names)),flush=True)
 df=pd.DataFrame(rows)
 (out/"errors.json").write_text(json.dumps(errors,indent=2))
 if df.empty:raise RuntimeError("No C4 sequential crossover candidates")
 df.to_csv(out/"candidates.csv",index=False)
 # Daily calendar is explicit from available market data, independent of C1.
 q=yf.download("QQQ",start="2022-01-01",end=END,auto_adjust=True,progress=False)
 calendar=[pd.Timestamp(x).tz_localize(None).normalize() for x in q.index]
 if not calendar:raise RuntimeError("Missing market calendar")
 scenarios={"C4_FULL_1":(1,1.),"C4_FOUR_25":(4,.25)}
 results={};ledgers=[];curves=[]
 for name,(slots,weight) in scenarios.items():
  perf,ledger,curve=simulate(df[["ticker","entry_date","exit_date","entry_price","exit_price","edge","reason"]].to_dict("records"),calendar,prices,slots,weight)
  results[name]=perf
  for r in ledger:r["strategy"]=name
  for r in curve:r["strategy"]=name
  ledgers+=ledger;curves+=curve
 pd.DataFrame(ledgers).to_csv(out/"ledger.csv",index=False)
 pd.DataFrame(curves).to_csv(out/"daily_equity.csv",index=False)
 summary={"status":"HISTORICAL_C4_EXPLORATORY_NOT_OOS","version":"C4-1",
 "signals":len(df),"unique_tickers":int(df.ticker.nunique()),"coverage":len(prices),
 "strategies":results,"limitations":[
 "20, 50, 120, 200 SMA crosses are required on distinct sessions in that order within 60 sessions.",
 "200-day MA cross must have RVOL20>=1.5 and prior below-MA200 history.",
 "Same-day multi-MA jumps are excluded from this strict sequential definition.",
 "Next-session open entry; TP8 SL4 and 5-session horizon are inherited C1 exit hypotheses, not optimized for C4.",
 "Current-listed universe is not point-in-time; no independently held-out data.",
 "Daily OHLC conservative stop-first execution; no intraday verification.",
 "No automatic trading, live alerts, or strategy promotion."]}
 (out/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False))
 table="".join(f"<tr><td>{html.escape(k)}</td><td>{v['return']:.2%}</td><td>{v['mdd']:.2%}</td><td>{v['trades']}</td></tr>" for k,v in results.items())
 (out/"report.html").write_text("<!doctype html><html lang='ko'><meta charset='utf-8'><title>PRISM C4</title><style>body{background:#101a2b;color:#eaf2ff;font:15px system-ui;margin:30px}h1{color:#65e7ba}td,th{padding:12px;border-bottom:1px solid #345}</style><h1>PRISM C4 · Sequential MA Breakout</h1><p>20→50→120→200 SMA · historical research only</p><p>Signals: "+str(len(df))+"</p><table><tr><th>Portfolio</th><th>Return</th><th>MDD</th><th>Trades</th></tr>"+table+"</table></html>",encoding="utf-8")
 print(json.dumps({"status":"GENERATED","signals":len(df)}))
if __name__=="__main__":main()
