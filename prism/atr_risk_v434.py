"""PRISM v4.34: ATR 3.5/4/4.5 sensitivity, initial-risk stop, yearly audit.
Frozen VCP80. Historical research, not independent out-of-sample.
"""
import argparse,json,html
from pathlib import Path
import numpy as np,pandas as pd
from .atr_frontier_v433 import exit_atr,CUTOFF
from .trend_v48 import load_prices,portfolio,STOP_SLIPPAGE
from .trend_v47 import atr

def exit_with_risk(d,entry,k,initial_risk):
 e=d.index.get_loc(entry)
 entry_price=float(d.Open.iloc[e]);a=atr(d);peak=entry_price;stop=-np.inf
 if initial_risk:
  stop=entry_price*.95
 for j in range(e,len(d)):
  op,low,close=map(float,(d.Open.iloc[j],d.Low.iloc[j],d.Close.iloc[j]))
  if j>e and np.isfinite(stop):
   if op<=stop:return d.index[j],op,"gap_stop"
   if low<=stop:return d.index[j],stop*(1-STOP_SLIPPAGE),"stop_touch_stress"
  if j-e>=180 and j+1<len(d):return d.index[j+1],float(d.Open.iloc[j+1]),"max_hold"
  peak=max(peak,close)
  if pd.notna(a.iloc[j]):
   stop=max(stop,peak-k*float(a.iloc[j]))
 return d.index[-1],float(d.Close.iloc[-1]),"terminal_mark_to_market"

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--scored",default="results/prism-v420/scored_signals.csv");ap.add_argument("--out",default="results/prism-v434");args=ap.parse_args()
 out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
 df=pd.read_csv(args.scored)
 for c in ("date","entry_date","exit_date"):df[c]=pd.to_datetime(df[c]).dt.normalize()
 if df.empty or df.rank_VCP_ALL.isna().any():raise RuntimeError("Missing frozen VCP scores")
 prices={t:load_prices(t,g.date.min()).loc[lambda x:x.index<=CUTOFF] for t,g in df.groupby("ticker")}
 modes={"MA30":None,"ATR3.5":(3.5,False),"ATR4":(4.,False),"ATR4.5":(4.5,False),"ATR4_RISK5":(4.,True)}
 report={"version":"v4.34","status":"HISTORICAL SENSITIVITY / NOT INDEPENDENT OOS","cutoff":str(CUTOFF.date()),"variants":{},
 "limitations":["Initial -5% risk stop is based on entry open; enforced only from following session due to daily-bar ambiguity.",
 "ATR trailing stop ratchets upward only, using completed closes.",
 "Gap stop executes at observed open; intraday touch applies adverse 20bps stress.",
 "Terminal marks are not realized sales, but legacy portfolio engine books them as synthetic liquidation.",
 "Yearly equity returns are calendar-year changes in marked equity, not sums of realized trades.",
 "Historical universe and ATR multipliers have been selected after observing historical outcomes; no independent OOS.",
 "Daily-close MDD excludes intraday troughs."]}
 all_ledger=[];all_curve=[];all_trades=[]
 for name,mode in modes.items():
  rows=[]
  for r in df.itertuples(index=False):
   d=prices[r.ticker]
   if r.entry_date not in d.index:raise RuntimeError("Missing entry "+r.ticker)
   if mode is None:xd,px,reason=r.exit_date,float(r.exit_price),r.reason
   else:xd,px,reason=exit_with_risk(d,r.entry_date,*mode)
   if xd<r.entry_date:raise RuntimeError("Bad exit order")
   rows.append({"ticker":r.ticker,"mode":name,"entry_date":str(r.entry_date.date()),"exit_date":str(xd.date()),
   "entry_price":float(r.entry_price),"exit_price":float(px),"edge":float(r.rank_VCP_ALL),"reason":reason})
  perf,ledger,curve=portfolio(rows,prices,name,rotation=False)
  if not ledger:raise RuntimeError("Zero trades "+name)
  lookup={(r["ticker"],r["entry_date"]):r for r in rows}
  prior=10_000_000.;trade_attribution=[];realized=0.;terminal=0.
  for x in ledger:
   z=lookup[(x["ticker"],x["entry_date"])]
   pnl=float(x["capital_after"])-prior
   mark=z["reason"]=="terminal_mark_to_market"
   if mark:terminal+=pnl
   else:realized+=pnl
   trade_attribution.append({**x,"strategy":name,"pnl_krw":pnl,"terminal_mark":mark})
   prior=float(x["capital_after"])
  win=sorted([x["pnl_krw"] for x in trade_attribution if x["pnl_krw"]>0],reverse=True)
  gross=sum(win)
  yearly={}
  for yr,g in pd.DataFrame(curve).assign(year=lambda x:x.date.str[:4]).groupby("year"):
   start=10_000_000. if yr==min(x["date"][:4] for x in curve) else prev
   end=float(g.equity.iloc[-1]);yearly[yr]=end/start-1;prev=end
  report["variants"][name]={**perf,"terminal_executed":sum(x["terminal_mark"] for x in trade_attribution),
   "realized_trade_pnl_krw":realized,"terminal_mark_pnl_krw":terminal,
   "top1_gross_win_share":sum(win[:1])/gross if gross else None,
   "top3_gross_win_share":sum(win[:3])/gross if gross else None,
   "top5_gross_win_share":sum(win[:5])/gross if gross else None,
   "yearly_marked_equity_return":yearly}
  all_trades+=trade_attribution
  for x in ledger:x["strategy"]=name
  for x in curve:x["strategy"]=name
  all_ledger+=ledger;all_curve+=curve
 pd.DataFrame(all_trades).to_csv(out/"trade_attribution.csv",index=False)
 pd.DataFrame(all_ledger).to_csv(out/"ledger.csv",index=False)
 pd.DataFrame(all_curve).to_csv(out/"daily_equity.csv",index=False)
 (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
 trs="".join(f"<tr><td>{html.escape(k)}</td><td>{v['return']:.2%}</td><td>{v['daily_close_mdd']:.2%}</td><td>{v['trades']}</td><td>{v['terminal_executed']}</td></tr>" for k,v in report["variants"].items())
 (out/"report.html").write_text("<!doctype html><meta charset='utf-8'><title>PRISM v4.34</title><style>body{background:#0b1729;color:#edf7ff;font:15px system-ui;margin:28px}h1{color:#5de4bd}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #345}</style><h1>PRISM v4.34 · ATR Sensitivity and Risk Audit</h1><p>Historical research only; terminal marks are not realized sales.</p><table><tr><th>Mode</th><th>Return</th><th>MDD</th><th>Trades</th><th>Terminal</th></tr>"+trs+"</table>",encoding="utf-8")
 print(json.dumps({"status":"generated","returns":{k:v["return"] for k,v in report["variants"].items()}},ensure_ascii=False))
if __name__=="__main__":main()
