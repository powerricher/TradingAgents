"""Frozen-entry exit comparison. No intraday fills are invented.
Requires selected_rows from v4.2 report.json; downloads daily OHLC for exits.
"""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,yfinance as yf

def exit_trade(df,idx,mode,max_days=10):
 entry=float(df.iloc[idx]["Close"]); previous=entry; peak=entry
 for j in range(idx+1,min(idx+max_days+1,len(df))):
  r=df.iloc[j]; o,h,l,c=[float(r[k]) for k in ("Open","High","Low","Close")]
  if mode=="A":return j,c,"next_close"
  if o<=previous*.99: return j,o,"gap_open"
  if mode=="B":
   if c<o:return j,c,"red_candle"
  if mode=="C":
   ma5=float(df["Close"].iloc[max(0,j-4):j+1].mean())
   # End-of-day trailing logic only; no impossible same-bar stop fill.
   if c<ma5 or c<peak*.95:
    if j+1<len(df): return j+1,float(df.iloc[j+1]["Open"]),"trend_break_next_open"
    return None,None,"unresolved_next_open"
  peak=max(peak,h);previous=c
  if j==idx+max_days:return j,c,"time_exit"
 return None,None,"no_future_bar"

def main():
 p=argparse.ArgumentParser();p.add_argument("--source",default="results/prism-v42/report.json")
 p.add_argument("--out",default="results/prism-exit");p.add_argument("--max-days",type=int,default=10)
 a=p.parse_args();src=json.loads(Path(a.source).read_text());signals=src["selected_rows"]
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);rows=[];errors=[]
 for ticker in sorted({r["ticker"] for r in signals}):
  z=[r for r in signals if r["ticker"]==ticker]
  try:
   d=yf.download(ticker,start="2021-01-01",end="2026-11-01",auto_adjust=False,progress=False)
   if isinstance(d.columns,pd.MultiIndex):d.columns=d.columns.get_level_values(0)
   d.index=pd.to_datetime(d.index).tz_localize(None).normalize()
   for s in z:
    dt=pd.Timestamp(s["date"]).normalize()
    if dt not in d.index:errors.append([ticker,str(dt),"missing entry"]);continue
    i=d.index.get_loc(dt)
    for mode in "ABC":
     j,price,reason=exit_trade(d,i,mode,a.max_days)
     if j is None:errors.append([ticker,str(dt),"insufficient future bars"]);continue
     entry=float(d.iloc[i]["Close"])
     rows.append(dict(ticker=ticker,entry_date=str(dt.date()),exit_date=str(d.index[j].date()),
                      mode=mode,holding_days=int(j-i),entry=entry,exit=price,
                      ret=price/entry-1,reason=reason))
  except Exception as e:errors.append([ticker,"download",str(e)])
 x=pd.DataFrame(rows);x.to_csv(out/"trades.csv",index=False)
 summary={"models":{},"errors":errors,"note":"Exit-only comparison on frozen v4.2 selected entries; close-to-close returns, no intraday fill assumptions."}
 for m,g in x.groupby("mode"):
  r=g.ret.to_numpy();profit=r[r>0].sum();loss=-r[r<0].sum()
  summary["models"][m]={"n":len(g),"avg_return":float(r.mean()),"win_rate":float((r>0).mean()),
   "pf":float(profit/loss) if loss else None,"avg_holding_days":float(g.holding_days.mean()),
   "gross_pnl_krw":int(round(r.sum()*1e7))}
 (out/"summary.json").write_text(json.dumps(summary,indent=2))
 (out/"report.html").write_text("<meta charset='utf-8'><h1>PRISM Frozen Entry — Exit Comparison</h1><pre>"+json.dumps(summary,indent=2)+"</pre>")
 print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
