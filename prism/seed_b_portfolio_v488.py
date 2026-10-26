"""PRISM v4.88 — frozen VCP/BCH signal portfolio replay.
Historical, current-listed universe, not independent OOS.
"""
import json,html
from pathlib import Path
import numpy as np,pandas as pd,yfinance as yf
from .trend_v49 import exit_trade
from .seed_c_multi_position_v471 import simulate
from .atr_frontier_v433 import CUTOFF
START=10_000_000.
def main():
 out=Path("results/prism-v488");out.mkdir(parents=True,exist_ok=True)
 sig=pd.read_csv("results/prism-v487/signal_features.csv")
 required={"ticker","detector","signal_date","entry_date","entry_price","edge"}
 if sig.empty or not required.issubset(sig):raise RuntimeError("Missing v4.87 signals")
 sig=sig[sig.detector.isin(["VCP_BREAKOUT","BCH_CONFIRMED"])].copy()
 names=sorted(sig.ticker.unique())
 prices={};errors=[];candidates=[]
 for start in range(0,len(names),25):
  batch=names[start:start+25]
  try:
   raw=yf.download(batch,start="2021-01-01",end="2026-10-07",auto_adjust=True,group_by="ticker",threads=False,progress=False)
  except Exception as e:
   errors.extend({"ticker":t,"error":str(e)} for t in batch);continue
  for t in batch:
   try:
    d=(raw[t] if isinstance(raw.columns,pd.MultiIndex) else raw).dropna(subset=["Open","High","Low","Close"]).copy()
    d.index=pd.DatetimeIndex(d.index).tz_localize(None).normalize()
    if d.empty:raise ValueError("Empty history")
    prices[t]=d
    for r in sig[sig.ticker==t].itertuples(index=False):
     day=pd.Timestamp(r.signal_date)
     if day not in d.index:raise ValueError("Missing signal date "+str(day))
     x=exit_trade(d,day,"CONFIRM30_1")
     if x is None:raise ValueError("Unresolved exit "+str(day))
     e,ix,entry,px,reason=x
     if abs(float(entry)/float(r.entry_price)-1)>.001:
      raise ValueError("Entry snapshot mismatch "+str(day))
     candidates.append({"ticker":t,"detector":r.detector,"signal_date":r.signal_date,
      "entry_date":str(d.index[e].date()),"exit_date":str(d.index[ix].date()),
      "entry_price":float(entry),"exit_price":float(px),"reason":reason,
      "edge":float(r.edge)})
   except Exception as e:errors.append({"ticker":t,"error":str(e)})
  print("v4.88 processed",min(start+25,len(names)),"/",len(names),flush=True)
 (out/"coverage_errors.json").write_text(json.dumps(errors,ensure_ascii=False,indent=2))
 if errors:raise RuntimeError("Unresolved price or exit records: "+str(len(errors))+"; see coverage_errors.json")
 df=pd.DataFrame(candidates)
 if df.empty or len(df)!=len(sig):raise RuntimeError("Incomplete candidate coverage")
 if df.duplicated(["ticker","detector","entry_date"]).any():raise RuntimeError("Duplicate candidates")
 df.to_csv(out/"candidate_exits.csv",index=False)
 q=yf.download("QQQ",start="2022-01-01",end="2026-10-07",auto_adjust=True,progress=False)
 if isinstance(q.columns,pd.MultiIndex):q.columns=q.columns.get_level_values(0)
 calendar=list(pd.DatetimeIndex(q.index).tz_localize(None).normalize())
 if not calendar:raise RuntimeError("Missing QQQ calendar")
 results={};ledgers=[];curves=[];years=[]
 for det in ("VCP_BREAKOUT","BCH_CONFIRMED"):
  z=df[df.detector==det]
  for policy,slots,weight in (("ONE_100",1,1.),("FOUR_25",4,.25)):
   name=det+"_"+policy
   records=z[["ticker","entry_date","exit_date","entry_price","exit_price","reason","edge"]].to_dict("records")
   perf,ledger,curve=simulate(records,calendar,prices,slots,weight)
   results[name]={**perf,"candidate_count":len(z)}
   for x in ledger:x["strategy"]=name
   for x in curve:x["strategy"]=name
   ledgers.extend(ledger);curves.extend(curve)
   eq=pd.DataFrame(curve);eq.date=pd.to_datetime(eq.date)
   for year,g in eq.groupby(eq.date.dt.year):
    prev=eq[eq.date<g.date.min()]
    initial=float(prev.equity.iloc[-1]) if len(prev) else START
    years.append({"strategy":name,"year":int(year),"return":float(g.equity.iloc[-1]/initial-1)})
 pd.DataFrame(ledgers).to_csv(out/"ledger.csv",index=False)
 pd.DataFrame(curves).to_csv(out/"daily_equity.csv",index=False)
 pd.DataFrame(years).to_csv(out/"yearly_returns.csv",index=False)
 report={"version":"v4.88","status":"HISTORICAL_PORTFOLIO_NOT_OOS","strategies":results,
 "requested_signals":len(sig),"resolved_signals":len(df),"unique_tickers":len(names),
 "limitations":["2500-name current-listed universe, not point-in-time, delisted names absent.",
 "Frozen VCP/BCH entry and CONFIRM30_1 exit; no tuning on forward outcomes.",
 "Portfolio uses 10bps round-trip friction and historical OHLC stop assumptions.",
 "Capital occupancy and same-day exit handling follow v4.71 simulator.",
 "Historical repeated research is in-sample; not independent OOS.",
 "No Seed A 2500 expansion or shared A/B/C/D cash in this stage.",
 "Annual +200% is an aspiration, not forecast; no live trading."]}
 (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
 trs="".join("<tr><td>%s</td><td>%.2f%%</td><td>%.2f%%</td><td>%d</td></tr>"%(html.escape(k),100*v["return"],100*v["mdd"],v["trades"]) for k,v in results.items())
 (out/"report.html").write_text("<!doctype html><html lang='ko'><meta charset='utf-8'><title>PRISM v4.88</title><style>body{background:#0b1729;color:#eaf4ff;font:15px system-ui;margin:30px}h1{color:#5de4bd}td,th{padding:10px;border-bottom:1px solid #345}</style><h1>PRISM v4.88 · Seed B 2500 Portfolio Replay</h1><p>Historical, not independent OOS or shared cash.</p><table><tr><th>Strategy</th><th>Return</th><th>MDD</th><th>Trades</th></tr>"+trs+"</table></html>",encoding="utf-8")
 print(json.dumps({"status":"V488_COMPLETE","strategies":results}))
if __name__=="__main__":main()
