"""Execution-corrected trade simulation for frozen close-generated signals.
D+1 open entry, C trend break confirmed at close then next open exit.
For time stop, exit at the next available open; no intrabar stop assumptions.
"""
import argparse,json
from pathlib import Path
import pandas as pd,numpy as np,yfinance as yf
from .universe_v45 import UNIVERSE
def main():
 p=argparse.ArgumentParser();p.add_argument("--source",default="results/prism-v45/report.json");p.add_argument("--out",default="results/prism-v45-execution");a=p.parse_args()
 src=json.loads(Path(a.source).read_text());out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 signals=pd.DataFrame(src["selected_rows"]);trades=[];issues=[]
 for ticker,g in signals.groupby("ticker"):
  try:
   d=yf.download(ticker,start="2021-01-01",end="2026-11-01",auto_adjust=True,progress=False)
   if isinstance(d.columns,pd.MultiIndex):d.columns=d.columns.get_level_values(0)
   d.index=pd.to_datetime(d.index).tz_localize(None).normalize()
   for s in g.itertuples():
    dt=pd.Timestamp(s.date).normalize()
    if dt not in d.index:issues.append([ticker,str(dt),"signal date missing"]);continue
    ix=d.index.get_loc(dt)
    if ix+2>=len(d):issues.append([ticker,str(dt),"no complete future bars"]);continue
    entry_ix=ix+1;entry=float(d.iloc[entry_ix].Open)
    for mode in "ABC":
     last=float(d.iloc[ix].Close);peak=entry;ex=None;reason=None
     for j in range(entry_ix,min(entry_ix+10,len(d)-1)+1):
      bar=d.iloc[j];op=float(bar.Open);close=float(bar.Close)
      if j>entry_ix and op<=last*.99:ex=op;exix=j;reason="gap_open";break
      if mode=="A":
       ex=close;exix=j;reason="first_close";break
      if mode=="B" and close<op:
       ex=float(d.iloc[j+1].Open);exix=j+1;reason="red_candle_next_open";break
      if mode=="C":
       ma5=float(d.Close.iloc[max(0,j-4):j+1].mean())
       if close<ma5 or close<peak*.95:
        ex=float(d.iloc[j+1].Open);exix=j+1;reason="trend_break_next_open";break
      peak=max(peak,float(bar.High));last=close
      if j-entry_ix>=9:
       ex=float(d.iloc[j+1].Open);exix=j+1;reason="time_exit_next_open";break
     if ex is None:issues.append([ticker,str(dt),mode+" unresolved"]);continue
     trades.append(dict(ticker=ticker,signal_date=str(dt.date()),entry_date=str(d.index[entry_ix].date()),exit_date=str(d.index[exix].date()),mode=mode,entry=entry,exit=ex,ret=ex/entry-1,holding_sessions=int(exix-entry_ix),reason=reason,edge=float(s.edge)))
  except Exception as e:issues.append([ticker,"download",str(e)])
 x=pd.DataFrame(trades);x.to_csv(out/"trades.csv",index=False)
 summary={"status":"EXPERIMENTAL","note":"No retraining of multi-horizon model; next-open execution correction only","signals":len(signals),"rows":len(x),"issues":issues,"strategies":{}}
 if not x.empty:
  for mode,g in x.groupby("mode"):
   r=g.ret.to_numpy();summary["strategies"][mode]={"n":len(g),"mean_gross":float(r.mean()),"mean_net_10bps":float(r.mean()-.001),"win_rate":float((r>0).mean()),"pf_gross":float(r[r>0].sum()/-r[r<0].sum()) if (r<0).any() else None}
 (out/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False))
 (out/"report.html").write_text("<meta charset='utf-8'><h1>PRISM v4.5 execution corrected</h1><pre>"+json.dumps(summary,indent=2,ensure_ascii=False)+"</pre>")
 print(json.dumps({k:v for k,v in summary.items() if k!="issues"},indent=2))
if __name__=="__main__":main()
