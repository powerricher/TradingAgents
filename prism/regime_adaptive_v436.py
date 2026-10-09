"""PRISM v4.36 regime-adaptive exit research. Historical, not OOS.
QQQ regime available at previous close only. Single position, no rotation.
"""
import argparse,json,html
from pathlib import Path
import numpy as np,pandas as pd,yfinance as yf
from .trend_v48 import load_prices,portfolio,STOP_SLIPPAGE
from .trend_v47 import atr
from .atr_frontier_v433 import exit_atr,CUTOFF

def regimes(first):
 q=yf.download("QQQ",start=(pd.Timestamp(first)-pd.Timedelta(days=440)).strftime("%Y-%m-%d"),end="2026-10-07",auto_adjust=True,progress=False)
 if isinstance(q.columns,pd.MultiIndex):q.columns=q.columns.get_level_values(0)
 if q.empty:raise RuntimeError("QQQ unavailable")
 q.index=pd.DatetimeIndex(q.index).tz_localize(None).normalize()
 close=q.Close.astype(float);ma=close.rolling(200,min_periods=200).mean();r=close.pct_change(60)
 observed=pd.Series(np.select([(close>ma)&(r>.05),(close<ma)&(r<-.05)],["BULL","BEAR"],default="SIDEWAYS"),index=q.index)
 observed.loc[ma.isna()|r.isna()]="UNKNOWN"
 return observed.shift(1).dropna()

def adaptive_exit(d,entry,regime,mode):
 e=d.index.get_loc(entry);a=atr(d);peak=float(d.Open.iloc[e]);stop=-np.inf
 fixed=regime.loc[entry]
 if fixed=="UNKNOWN":raise RuntimeError("Missing regime history")
 for j in range(e,len(d)):
  day=d.index[j];op=float(d.Open.iloc[j]);low=float(d.Low.iloc[j]);close=float(d.Close.iloc[j])
  active=fixed if mode=="FIXED" else regime.get(day,"UNKNOWN")
  if active=="UNKNOWN":raise RuntimeError("Regime missing "+str(day))
  if j>e and np.isfinite(stop) and (mode=="DYNAMIC" or fixed!="BULL"):
   if op<=stop:return day,op,"gap_atr_stop"
   if low<=stop:return day,stop*(1-STOP_SLIPPAGE),"atr_touch_stress"
  if j-e>=180 and j+1<len(d):return d.index[j+1],float(d.Open.iloc[j+1]),"max_hold"
  if active=="BULL":
   ma30=float(d.Close.iloc[max(0,j-29):j+1].mean())
   if close<ma30 and j+1<len(d):
    return d.index[j+1],float(d.Open.iloc[j+1]),"bull_ma30_next_open"
  peak=max(peak,close)
  if pd.notna(a.iloc[j]):stop=max(stop,peak-4*float(a.iloc[j]))
 return d.index[-1],float(d.Close.iloc[-1]),"terminal_mark_to_market"

def main():
 p=argparse.ArgumentParser();p.add_argument("--scored",default="results/prism-v420/scored_signals.csv");p.add_argument("--out",default="results/prism-v436");z=p.parse_args()
 out=Path(z.out);out.mkdir(parents=True,exist_ok=True)
 df=pd.read_csv(z.scored)
 for c in ("date","entry_date","exit_date"):df[c]=pd.to_datetime(df[c]).dt.normalize()
 if df.empty or df.rank_VCP_ALL.isna().any():raise RuntimeError("Missing frozen signals")
 prices={t:load_prices(t,g.date.min()).loc[lambda x:x.index<=CUTOFF] for t,g in df.groupby("ticker")}
 regime=regimes(df.entry_date.min())
 modes=("MA30_CONTROL","ATR4_CONTROL","ADAPTIVE_FIXED","ADAPTIVE_DYNAMIC","ADAPTIVE_FIXED_BEAR_CASH")
 report={"version":"v4.36","status":"EXPLORATORY / NOT INDEPENDENT OOS","strategies":{},
 "notes":["Prior-close QQQ regime only; bull above MA200 and 60D >5%, bear below MA200 and 60D <-5%, else sideways.",
 "Fixed: entry regime selects bull MA30-close breach vs other ATR4, no later change.",
 "Dynamic: bull MA30-close breach and nonbull ATR4; ATR stop ratchets every day, executes only in nonbull regimes.",
 "Original MA30 control retains frozen exit records and is not the same as the simplified adaptive MA30 exit.",
 "Bear cash blocks NEW entries only, never forces sale of an existing holding.",
 "ATR stop touches assume 20bps slippage, gaps fill at observed open; MA30 signal fills next open.",
 "Terminal mark-to-market is synthetic liquidation in legacy portfolio engine, not a realized sale.",
 "Regime-specific outcomes were inspected in earlier experiments; no independent OOS and universe survivorship bias.",
 "Daily-close MDD excludes intraday drawdowns."]}
 ledgers=[];curves=[];candidates=[]
 for mode in modes:
  rows=[]
  for r in df.itertuples(index=False):
   d=prices[r.ticker];entry=r.entry_date
   if entry not in d.index or entry not in regime.index:raise RuntimeError("Missing entry/regime "+r.ticker)
   state=regime.loc[entry]
   if state=="UNKNOWN":raise RuntimeError("Unknown regime at "+str(entry))
   if mode=="ADAPTIVE_FIXED_BEAR_CASH" and state=="BEAR":continue
   if mode=="MA30_CONTROL":day,px,reason=r.exit_date,float(r.exit_price),r.reason
   elif mode=="ATR4_CONTROL":day,px,reason=exit_atr(d,entry,4)
   else:day,px,reason=adaptive_exit(d,entry,regime,"DYNAMIC" if mode=="ADAPTIVE_DYNAMIC" else "FIXED")
   if day<entry:raise RuntimeError("Invalid exit order")
   rows.append({"ticker":r.ticker,"mode":mode,"entry_date":str(entry.date()),"exit_date":str(day.date()),
    "entry_price":float(r.entry_price),"exit_price":float(px),"edge":float(r.rank_VCP_ALL),"reason":reason,
    "entry_regime":state})
  perf,ledger,curve=portfolio(rows,prices,mode,rotation=False)
  if not ledger:raise RuntimeError("Zero-trade strategy "+mode)
  lookup={(r["ticker"],r["entry_date"]):r for r in rows}
  terminal=[x for x in ledger if lookup[(x["ticker"],x["entry_date"])]["reason"]=="terminal_mark_to_market"]
  report["strategies"][mode]={**perf,"terminal_executed":len(terminal),"candidate_signals":len(rows),
    "executed_entry_regimes":pd.Series([lookup[(x["ticker"],x["entry_date"])]["entry_regime"] for x in ledger]).value_counts().to_dict()}
  for r in rows:r["variant"]=mode
  for r in ledger:r["variant"]=mode
  for r in curve:r["variant"]=mode
  candidates+=rows;ledgers+=ledger;curves+=curve
 pd.DataFrame(candidates).to_csv(out/"candidate_exits.csv",index=False)
 pd.DataFrame(ledgers).to_csv(out/"ledger.csv",index=False)
 pd.DataFrame(curves).to_csv(out/"daily_equity.csv",index=False)
 (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
 trs="".join(f"<tr><td>{html.escape(k)}</td><td>{v['return']:.2%}</td><td>{v['daily_close_mdd']:.2%}</td><td>{v['trades']}</td><td>{v['terminal_executed']}</td></tr>" for k,v in report["strategies"].items())
 (out/"report.html").write_text("<!doctype html><meta charset='utf-8'><title>PRISM v4.36</title><style>body{background:#0b1729;color:#edf7ff;font:15px system-ui;margin:28px}h1{color:#5de4bd}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #345}</style><h1>PRISM v4.36 · Regime Adaptive Exit</h1><p>VCP80 · one slot · QQQ prior-close regimes · historical research</p><table><tr><th>Strategy</th><th>Return</th><th>Daily MDD</th><th>Trades</th><th>Terminal marks</th></tr>"+trs+"</table>",encoding="utf-8")
 print(json.dumps({"status":"generated","strategies":{k:v["return"] for k,v in report["strategies"].items()}},ensure_ascii=False))
if __name__=="__main__":main()
