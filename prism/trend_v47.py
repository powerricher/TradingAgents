"""PRISM v4.7 trend capture: frozen v4.5 entries, daily OHLC exit-only experiment."""
import argparse,json,html
from pathlib import Path
import numpy as np,pandas as pd,yfinance as yf
from .portfolio_v46 import calc,stats
CONFIG={"D":{"max_days":30},"E":{"max_days":60},"F":{"max_days":60}}
def atr(d,n=14):
 prev=d.Close.shift(1)
 tr=pd.concat([(d.High-d.Low).abs(),(d.High-prev).abs(),(d.Low-prev).abs()],axis=1).max(axis=1)
 return tr.rolling(n,min_periods=n).mean()
def simulate(d,signal_idx,mode):
 entry_idx=signal_idx+1
 if entry_idx>=len(d):return None
 entry=float(d.Open.iloc[entry_idx])
 if not np.isfinite(entry) or entry<=0:return None
 a=atr(d);maxdays=CONFIG[mode]["max_days"];peak=entry;stop=-np.inf
 # Stop decisions based on PREVIOUS close; exit at following open.
 for j in range(entry_idx,min(entry_idx+maxdays,len(d))):
  if j>entry_idx and np.isfinite(stop) and float(d.Open.iloc[j])<stop:
   return (entry_idx,j,entry,float(d.Open.iloc[j]),"prior_close_stop_gap")
  if j>entry_idx and np.isfinite(stop) and float(d.Low.iloc[j])<=stop:
   # stop was established at prior close, so daily OHLC can support touch,
   # but fills at stop are optimistic under slippage; stress separately.
   return (entry_idx,j,entry,stop,"prior_close_stop_touch")
  close=float(d.Close.iloc[j]);peak=max(peak,close)
  if j-entry_idx+1>=maxdays:
   if j+1<len(d):return(entry_idx,j+1,entry,float(d.Open.iloc[j+1]),"time_exit_next_open")
   return None
  ma10=float(d.Close.iloc[max(0,j-9):j+1].mean())
  ma20=float(d.Close.iloc[max(0,j-19):j+1].mean())
  av=float(a.iloc[j]) if pd.notna(a.iloc[j]) else np.nan
  if mode=="D":
   if j>entry_idx and close<ma10 and close<float(d.Close.iloc[j-1]):
    if j+1<len(d):return(entry_idx,j+1,entry,float(d.Open.iloc[j+1]),"trend_break_next_open")
    return None
  elif mode=="E":
   if np.isfinite(av):stop=max(stop,peak-3*av)
   if close<ma20:
    if j+1<len(d):return(entry_idx,j+1,entry,float(d.Open.iloc[j+1]),"ma20_break_next_open")
    return None
  else:
   ma20prev=float(d.Close.iloc[max(0,j-20):j].mean()) if j>=20 else ma20
   strong=close>ma20 and ma20>ma20prev
   k=4 if strong else (3 if close>=ma20 else 2)
   if np.isfinite(av):stop=max(stop,peak-k*av)
   if close<ma20 and j>entry_idx and float(d.Close.iloc[j-1])<ma20prev:
    if j+1<len(d):return(entry_idx,j+1,entry,float(d.Open.iloc[j+1]),"two_close_trend_break")
    return None
 return None
def main():
 p=argparse.ArgumentParser();p.add_argument("--signals",default="results/prism-v45/report.json");p.add_argument("--baseline",default="results/prism-v45-execution/trades.csv");p.add_argument("--out",default="results/prism-v47");a=p.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 selected=pd.DataFrame(json.loads(Path(a.signals).read_text())["selected_rows"])
 base=pd.read_csv(a.baseline);rows=[];issues=[]
 for ticker,g in selected.groupby("ticker"):
  try:
   first=pd.Timestamp(g.date.min())-pd.Timedelta(days=100)
   d=yf.download(ticker,start=str(first.date()),end="2026-11-01",auto_adjust=True,progress=False)
   if isinstance(d.columns,pd.MultiIndex):d.columns=d.columns.get_level_values(0)
   d.index=pd.to_datetime(d.index).tz_localize(None).normalize()
   for s in g.itertuples(index=False):
    dt=pd.Timestamp(s.date).normalize()
    if dt not in d.index:issues.append([ticker,str(dt),"missing_signal"]);continue
    i=d.index.get_loc(dt)
    for mode in "DEF":
     z=simulate(d,i,mode)
     if z is None:issues.append([ticker,str(dt),mode+" unresolved"]);continue
     e,x,entry,exit,reason=z
     rows.append(dict(ticker=ticker,signal_date=str(dt.date()),entry_date=str(d.index[e].date()),exit_date=str(d.index[x].date()),mode=mode,entry=entry,exit=exit,ret=exit/entry-1,holding_sessions=x-e,reason=reason,edge=float(s.edge)))
  except Exception as e:issues.append([ticker,"download",str(e)])
 new=pd.DataFrame(rows)
 if new.empty:raise RuntimeError("No trend trades")
 alltr=pd.concat([base,new],ignore_index=True)
 results={"version":"v4.7","status":"CHALLENGER ONLY","frozen_signals":len(selected),"cost_bps":10,"issues":issues,"strategies":{},"cautions":["Historical 50-stock universe is not point-in-time.","OHLC stop-touch assumes stop fill with zero stop slippage; gaps fill at open.","Closed-trade MDD omits unrealized drawdowns.","Longer holding periods block overlapping signals.","Same historical sample used to develop exit strategies; out-of-sample validation needed."]}
 ledger=[]
 for mode in "ABCDEF":
  g=alltr[alltr["mode"]==mode]
  if len(g)!=len(selected):results["cautions"].append(mode+" incomplete sample "+str(len(g)))
  portfolio,logs=calc(alltr,mode,cost=.001)
  results["strategies"][mode]={"portfolio":portfolio,"trade_statistics":stats(g,.001),"mean_holding":float(g.holding_sessions.mean()) if "holding_sessions" in g else None}
  for r in logs:r["mode"]=mode
  ledger+=logs
 alltr.to_csv(out/"trades.csv",index=False);pd.DataFrame(ledger).to_csv(out/"portfolio_ledger.csv",index=False)
 (out/"summary.json").write_text(json.dumps(results,ensure_ascii=False,indent=2))
 table="".join(f"<tr><td>{m}</td><td>{v['portfolio']['ending_capital']:,.0f}</td><td>{v['portfolio']['total_return']:.2%}</td><td>{v['portfolio']['closed_trade_mdd']:.2%}</td><td>{v['portfolio']['trades']}</td></tr>" for m,v in results["strategies"].items())
 (out/"report.html").write_text("<meta charset='utf-8'><style>body{background:#0b1425;color:#e9f2ff;font:15px system-ui;padding:25px}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #38506b}h1{color:#65e5bf}</style><h1>PRISM v4.7 Trend Capture</h1><p>50 stocks / frozen entries / 10bps costs / experimental</p><table><tr><th>Strategy</th><th>Final KRW</th><th>Return</th><th>Closed MDD</th><th>Trades</th></tr>"+table+"</table><pre>"+html.escape(json.dumps(results,ensure_ascii=False,indent=2))+"</pre>")
 print(json.dumps({"status":"complete","strategies":results["strategies"],"issue_count":len(issues)},ensure_ascii=False))
if __name__=="__main__":main()
