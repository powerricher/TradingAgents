"""PRISM v4.32: fixed VCP80 signal selection, alternative exit frontier.
Daily OHLC backtest, signal at close / entry next open; research only.
"""
import argparse,json,html
from pathlib import Path
import numpy as np,pandas as pd
from .trend_v48 import load_prices,portfolio,STOP_SLIPPAGE
from .trend_v47 import atr
MODES=("MA30","MA50","MA60_CONFIRM2","ATR4","HYBRID")
def exit_variant(d,entry,baseline_exit,baseline_price,mode):
 if mode=="MA30":return baseline_exit,baseline_price,"original_ma30"
 e=d.index.get_loc(entry)
 entry_px=float(d.Open.iloc[e])
 av=atr(d);peak=entry_px;below=0
 max_hold=180
 for j in range(e,min(len(d)-1,e+max_hold)+1):
  op=float(d.Open.iloc[j]);lo=float(d.Low.iloc[j]);cl=float(d.Close.iloc[j])
  if j>e:
   if np.isfinite(stop) and op<=stop:return d.index[j],op,"gap_stop"
   if np.isfinite(stop) and lo<=stop:return d.index[j],stop*(1-STOP_SLIPPAGE),"stop_touch_stress"
  if j-e>=max_hold and j+1<len(d):return d.index[j+1],float(d.Open.iloc[j+1]),"max_hold"
  # MA signals observed only at today's close, execute following session open.
  window=50 if mode=="MA50" else 60
  ma=float(d.Close.iloc[max(0,j-window+1):j+1].mean())
  if mode in ("MA50","MA60_CONFIRM2"):
   below=below+1 if cl<ma else 0
   if below>=(2 if mode=="MA60_CONFIRM2" else 1):
    if j+1<len(d):return d.index[j+1],float(d.Open.iloc[j+1]),"ma_exit"
  if mode=="HYBRID":
   # Entry failure protection until +10% achieved on a completed close.
   if peak<entry_px*1.10:
    if cl<=entry_px*.95 and j+1<len(d):return d.index[j+1],float(d.Open.iloc[j+1]),"early_loss_next_open"
   else:
    below=below+1 if cl<ma else 0
    if below>=2 and j+1<len(d):return d.index[j+1],float(d.Open.iloc[j+1]),"hybrid_ma60"
  peak=max(peak,cl)
  avj=float(av.iloc[j]) if pd.notna(av.iloc[j]) else np.nan
  if mode=="ATR4":
   stop=peak-4*avj if np.isfinite(avj) else -np.inf
  elif mode=="HYBRID":
   stop=(entry_px*.95 if peak<entry_px*1.10 else peak-4*avj) if np.isfinite(avj) else entry_px*.95
  else:stop=-np.inf
 return None
def main():
 p=argparse.ArgumentParser();p.add_argument("--scored",default="results/prism-v420/scored_signals.csv");p.add_argument("--out",default="results/prism-v432");a=p.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 df=pd.read_csv(a.scored)
 for c in ("date","entry_date","exit_date"):df[c]=pd.to_datetime(df[c]).dt.normalize()
 prices={t:load_prices(t,g.date.min()) for t,g in df.groupby("ticker")}
 # Shared historical evaluation cutoff; no strategy may use later prices.
 horizon=pd.Timestamp("2026-10-06")
 for ticker,d in prices.items():
  prices[ticker]=d.loc[d.index<=horizon].copy()
  if prices[ticker].empty:raise RuntimeError("Missing evaluation history "+ticker)
 if df.empty or df.rank_VCP_ALL.isna().any():raise RuntimeError("Missing frozen VCP signals")
 results={"version":"v4.32","status":"HISTORICAL EXIT RESEARCH / NOT INDEPENDENT OOS","strategies":{},"trade_level":{},
 "notes":["Same VCP80 signal ranks for all exit variants; no model retraining.",
 "Unresolved positions are valued at the common 2026-10-06 terminal close; this is a synthetic portfolio liquidation, not a real sell signal.",
 "Exit decisions at close fill next open; gap stops fill at open; touch stops assume 20bps adverse slippage.",
 "Entry-day intraday stops are not assumed executable due to unknown OHLC sequence.",
 "Alternative exits can hold up to 180 sessions, unlike the original MA30 max holding window.",
 "Same retrospectively selected 80-stock sample repeatedly researched; no independent OOS.",
 "Trade-level MFE/MAE exclude entry-day intraday extremes; profit capture ratio is undefined for nonpositive MFE.",
 "Daily-close MDD excludes intraday lows."]}
 results["evaluation_horizon"]="2026-10-06"
 ledgers=[];curves=[];trades=[]
 for mode in MODES:
  rows=[];trade_stats=[]
  for r in df.itertuples(index=False):
   d=prices[r.ticker];e=d.index.get_loc(r.entry_date)
   if mode=="MA30":outcome=(r.exit_date,float(r.exit_price),r.reason)
   else:outcome=exit_variant(d,r.entry_date,r.exit_date,float(r.exit_price),mode)
   if outcome is None:raise RuntimeError("Missing exit data "+r.ticker+" "+mode)
   xday,xprice,reason=outcome;x=d.index.get_loc(xday)
   if x<e:raise RuntimeError("Invalid exit order")
   highs=d.High.iloc[e+1:x].to_numpy(dtype=float)
   lows=d.Low.iloc[e+1:x].to_numpy(dtype=float)
   mfe=max(0,float(np.max(highs)/r.entry_price-1)) if len(highs) else 0.
   mae=min(0,float(np.min(lows)/r.entry_price-1)) if len(lows) else 0.
   realized=float(xprice/r.entry_price-1)
   trade_stats.append({"ticker":r.ticker,"entry_date":str(r.entry_date.date()),"exit_date":str(xday.date()),"mode":mode,
    "ret_before_cost":realized,"mfe":mfe,"mae":mae,"capture_ratio":realized/mfe if mfe>0 else np.nan,
    "giveback":mfe-realized,"holding_sessions":x-e,"reason":reason})
   rows.append({"ticker":r.ticker,"mode":mode,"entry_date":str(r.entry_date.date()),"exit_date":str(xday.date()),
    "entry_price":float(r.entry_price),"exit_price":float(xprice),"edge":float(r.rank_VCP_ALL),"reason":reason})
  perf,ledger,curve=portfolio(rows,prices,mode,rotation=False)
  if perf["trades"]==0:raise RuntimeError("Zero-trade portfolio "+mode)
  results["strategies"][mode]=perf
  results["strategies"][mode]["terminal_marks"]=int((st.reason=="terminal_mark_to_market").sum()) if "st" in locals() else sum(x["reason"]=="terminal_mark_to_market" for x in trade_stats)
  st=pd.DataFrame(trade_stats)
  results["trade_level"][mode]={"n":len(st),"mean_mfe":float(st.mfe.mean()),"mean_mae":float(st.mae.mean()),
   "mean_giveback":float(st.giveback.mean()),"mean_hold_sessions":float(st.holding_sessions.mean()),
   "mean_capture_ratio":float(st.capture_ratio.dropna().mean()) if st.capture_ratio.notna().any() else None}
  trades+=trade_stats
  for x in ledger:x["variant"]=mode
  for x in curve:x["variant"]=mode
  ledgers+=ledger;curves+=curve
 pd.DataFrame(trades).to_csv(out/"trade_exit_audit.csv",index=False)
 pd.DataFrame(ledgers).to_csv(out/"ledger.csv",index=False)
 pd.DataFrame(curves).to_csv(out/"daily_equity.csv",index=False)
 (out/"summary.json").write_text(json.dumps(results,ensure_ascii=False,indent=2))
 rows="".join(f"<tr><td>{html.escape(k)}</td><td>{v['return']:.2%}</td><td>{v['daily_close_mdd']:.2%}</td><td>{v['trades']}</td></tr>" for k,v in results["strategies"].items())
 (out/"report.html").write_text("<!doctype html><meta charset='utf-8'><title>PRISM v4.32</title><style>body{background:#0b1729;color:#edf7ff;font:15px system-ui;margin:28px}h1{color:#5de4bd}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #345}</style><h1>PRISM v4.32 · Exit Frontier & Hybrid Trend</h1><p>VCP80 frozen entries · initial KRW 10M · research only</p><table><tr><th>Exit</th><th>Return</th><th>Daily MDD</th><th>Trades</th></tr>"+rows+"</table>",encoding="utf-8")
 print(json.dumps({"status":"generated","strategies":{k:v["return"] for k,v in results["strategies"].items()}},ensure_ascii=False))
if __name__=="__main__":main()
