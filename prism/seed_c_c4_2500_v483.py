"""PRISM C4 v4.83: 2500-name sequential MA audit; strict and flexible research.
No LOPE-specific tuning; not independent OOS.
"""
import json,html
from pathlib import Path
import numpy as np,pandas as pd,yfinance as yf
from .seed_c_sequential_ma_c4 import signals as strict_signals
from .seed_c_sequential_ma import detect as flexible_detect
from .seed_c_execution_v468 import exit_path
from .seed_c_multi_position_v471 import simulate
END="2026-10-07"
def main():
 out=Path("results/prism-c4-v483");out.mkdir(parents=True,exist_ok=True)
 names=pd.read_csv("results/prism-v479/universe_snapshot.csv").ticker.astype(str).tolist()
 if len(names)!=2500:raise RuntimeError("2500-name snapshot required")
 q=yf.download("QQQ",start="2022-01-01",end=END,auto_adjust=True,progress=False)
 if isinstance(q.columns,pd.MultiIndex):q.columns=q.columns.get_level_values(0)
 calendar=list(pd.DatetimeIndex(q.index).tz_localize(None).normalize())
 if not calendar:raise RuntimeError("Missing market calendar")
 rows=[];prices={};errors=[]
 for j in range(0,len(names),25):
  batch=names[j:j+25]
  try:raw=yf.download(batch,start="2021-01-01",end=END,auto_adjust=True,group_by="ticker",threads=False,progress=False)
  except Exception as exc:
   errors.extend({"ticker":t,"error":str(exc)} for t in batch);continue
  for t in batch:
   try:
    d=(raw[t] if isinstance(raw.columns,pd.MultiIndex) else raw).dropna(subset=["Open","High","Low","Close","Volume"]).copy()
    d.index=pd.DatetimeIndex(d.index).tz_localize(None).normalize()
    if len(d)<260:continue
    prices[t]=d
    modes={"STRICT_60":[(i,rv) for i,stages,rv in strict_signals(d)],
           "FLEX_60":[(i,None) for i in flexible_detect(d,60,False)]}
    for mode,events in modes.items():
     for i,rv in events:
      if d.index[i]<pd.Timestamp("2022-01-01") or i+6>=len(d):continue
      ep=float(d.Open.iloc[i+1])
      if not np.isfinite(ep) or ep<=0:continue
      x=exit_path(d,d.index[i+1],ep,.08,.04,5)
      if x is None:continue
      xd,xp,reason=x
      volume=d.Volume.astype(float);prior=volume.iloc[i-20:i].mean()
      rvol=float(volume.iloc[i]/prior) if prior>0 else 0.
      if mode=="FLEX_60" and rvol<1.5:continue
      rows.append({"ticker":t,"mode":mode,"signal_date":str(d.index[i].date()),
       "entry_date":str(d.index[i+1].date()),"entry_price":ep,"exit_date":xd,
       "exit_price":xp,"edge":rvol,"reason":reason,"gross_return":xp/ep-1})
   except Exception as exc:errors.append({"ticker":t,"error":str(exc)})
  print("C4 v4.83 processed",min(j+25,len(names)),flush=True)
 df=pd.DataFrame(rows)
 (out/"coverage_errors.json").write_text(json.dumps(errors,indent=2))
 if df.empty:raise RuntimeError("No C4 candidates")
 if df.duplicated(["ticker","mode","entry_date"]).any():raise RuntimeError("Duplicate candidates")
 df.to_csv(out/"candidate_exits.csv",index=False)
 report={"version":"v4.83","status":"HISTORICAL_C4_CHALLENGER_NOT_OOS",
 "requested":2500,"loaded":len(prices),"strategies":{},
 "limitations":["Strict 20/50/120/200 distinct-session state-machine versus flexible 60-session prior crosses.",
 "Close signal, next open entry; TP8 SL4 H5 inherited from Seed C, not optimized for C4.",
 "Strict final 200 cross RVOL20 >=1.5; flexible same final RVOL requirement.",
 "Current-listed 2500-name universe has survivorship and delisting bias.",
 "No LOPE-specific parameter fitting, but repeated historical research is in-sample.",
 "Stop-first daily OHLC assumptions; no intraday fills, no shared ABCD cash.",
 "No automatic trading or promotion."]}
 ledgers=[];curves=[]
 for mode,z in df.groupby("mode"):
  records=z[["ticker","entry_date","entry_price","exit_date","exit_price","edge","reason"]].to_dict("records")
  perf,ledger,curve=simulate(records,calendar,prices,4,.25)
  report["strategies"][mode]={**perf,"candidate_count":len(z)}
  for x in ledger:x["mode"]=mode
  for x in curve:x["mode"]=mode
  ledgers+=ledger;curves+=curve
 pd.DataFrame(ledgers).to_csv(out/"ledger.csv",index=False)
 pd.DataFrame(curves).to_csv(out/"daily_equity.csv",index=False)
 (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
 trs="".join("<tr><td>%s</td><td>%d</td><td>%.2f%%</td><td>%.2f%%</td><td>%d</td></tr>"%(html.escape(k),v["candidate_count"],100*v["return"],100*v["mdd"],v["trades"]) for k,v in report["strategies"].items())
 (out/"report.html").write_text("<!doctype html><html lang='ko'><meta charset='utf-8'><title>PRISM C4 v4.83</title><style>body{background:#0b1729;color:#eaf4ff;font:15px system-ui;margin:30px}h1{color:#5de4bd}td,th{padding:10px;border-bottom:1px solid #345}</style><h1>PRISM C4 v4.83 · Sequential MA 2500</h1><p>Historical challenger, NOT OOS or production.</p><table><tr><th>Mode</th><th>Candidates</th><th>Return</th><th>MDD</th><th>Trades</th></tr>"+trs+"</table></html>",encoding="utf-8")
 print(json.dumps({"status":"C4_V483_COMPLETE","strategies":report["strategies"]}))
if __name__=="__main__":main()
