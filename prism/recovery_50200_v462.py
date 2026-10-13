"""PRISM v4.62: independent Recovery MA50/MA200 entry and stop frontier.
Exploratory historical backtest; Seed A/B frozen and unchanged.
"""
import json,html
from pathlib import Path
import numpy as np,pandas as pd
from .universe_v445 import UNIVERSE
from .trend_v48 import load_prices,STOP_SLIPPAGE
from .recovery_portfolio_v462_fix import portfolio_same_day
from .trend_v47 import atr
from .atr_frontier_v433 import CUTOFF
MODES=("BASIC","HIGHER_LOW","MOMENTUM")
EXITS=("STOP6","STOP7","STAGED7")
def detect(d,qqq,ticker):
 c=d.Close.astype(float);lo=d.Low.astype(float);hi=d.High.astype(float)
 ma50=c.rolling(50,min_periods=50).mean()
 ma200=c.rolling(200,min_periods=200).mean()
 ma50up=ma50>ma50.shift(10)
 near=(c/ma200>=.85)&(c/ma200<=.97)
 base=(c>ma50)&(c<ma200)&ma50up&near
 low_new=lo.rolling(20,min_periods=20).min()
 low_old=lo.shift(20).rolling(20,min_periods=20).min()
 defended=low_new>=low_old
 rs=c.pct_change(20)-qqq.pct_change(20).reindex(d.index)
 breakout=c>hi.shift(1).rolling(10,min_periods=10).max()
 conditions={"BASIC":base,"HIGHER_LOW":base&defended,
             "MOMENTUM":base&defended&(rs>0)&breakout}
 result=[]
 for mode,mask in conditions.items():
  # Prevent repeated daily signals from a single persistent setup:
  # first qualifying day after a 10-session nonqualifying interval.
  fresh=mask&(~mask.shift(1).rolling(10,min_periods=1).max().fillna(0).astype(bool))
  fresh=fresh.fillna(False)&(d.index>=pd.Timestamp("2022-01-01"))
  fresh=fresh&(d.index<=CUTOFF-pd.Timedelta(days=220))
  for dt in d.index[fresh]:
   i=d.index.get_loc(dt)
   if i+1>=len(d):continue
   result.append({"ticker":ticker,"mode":mode,"signal_date":str(dt.date()),
     "entry_date":str(d.index[i+1].date()),"entry_price":float(d.Open.iloc[i+1]),
     "edge":float((c.iloc[i]/ma50.iloc[i]-1)+max(0.,rs.iloc[i] if pd.notna(rs.iloc[i]) else 0.))})
 return result
def exit_rule(d,entry,price,mode):
 e=d.index.get_loc(pd.Timestamp(entry))
 av=atr(d)
 fixed=price*(.94 if mode=="STOP6" else .93)
 stop=fixed;peak=price;activated=False
 for j in range(e,len(d)):
  op=float(d.Open.iloc[j]);low=float(d.Low.iloc[j]);cl=float(d.Close.iloc[j])
  # Entry-session fixed risk protection is active; trailing updates
  # only after the close, never from the same-day high.
  if op<=stop:return str(d.index[j].date()),op,"gap_stop"
  if low<=stop:return str(d.index[j].date()),stop*(1-STOP_SLIPPAGE),"stop_touch_stress"
  if j-e>=120 and j+1<len(d):
   return str(d.index[j+1].date()),float(d.Open.iloc[j+1]),"max_hold"
  if mode=="STAGED7":
   peak=max(peak,cl)
   if peak>=price*1.08:activated=True
   if activated:
    stop=max(stop,price)
    if pd.notna(av.iloc[j]):stop=max(stop,peak-3*float(av.iloc[j]))
  # Fixed stop modes have no profit-taking except the 120-session limit.
 return str(d.index[-1].date()),float(d.Close.iloc[-1]),"terminal_mark_to_market"
def main():
 out=Path("results/prism-v462");out.mkdir(parents=True,exist_ok=True)
 prices={};errors=[]
 for t in UNIVERSE:
  try:
   d=load_prices(t,"2020-01-01").loc[lambda x:x.index<=CUTOFF]
   if len(d)<220:raise ValueError("Insufficient bars")
   prices[t]=d
  except Exception as e:errors.append({"ticker":t,"error":str(e)})
 if len(prices)<80:raise RuntimeError("Insufficient market coverage")
 q=load_prices("QQQ","2020-01-01").loc[lambda x:x.index<=CUTOFF].Close
 signals=[]
 for t,d in prices.items():signals.extend(detect(d,q,t))
 if not signals:raise RuntimeError("No Recovery candidates")
 rows=[];post=[]
 for r in signals:
  d=prices[r["ticker"]]
  for mode in EXITS:
   day,px,reason=exit_rule(d,r["entry_date"],r["entry_price"],mode)
   rows.append({**r,"exit_mode":mode,"exit_date":day,"exit_price":px,"reason":reason,
    "gross_return":px/r["entry_price"]-1})
   if reason in ("gap_stop","stop_touch_stress"):
    idx=d.index.get_loc(pd.Timestamp(day))
    if idx+20<len(d):
     post.append({"ticker":r["ticker"],"mode":r["mode"],"exit_mode":mode,
      "entry_date":r["entry_date"],"exit_date":day,
      "post_stop_20d_return":float(d.Close.iloc[idx+20]/px-1)})
 candidates=pd.DataFrame(rows)
 candidates.to_csv(out/"candidate_exits.csv",index=False)
 pd.DataFrame(signals).to_csv(out/"entry_signals.csv",index=False)
 pd.DataFrame(post).to_csv(out/"post_stop_rebounds.csv",index=False)
 (out/"coverage_errors.json").write_text(json.dumps(errors,indent=2))
 report={"version":"v4.62","status":"EXPLORATORY RECOVERY 50/200 / NOT OOS",
  "coverage":{"requested":150,"available":len(prices),"excluded":errors},
  "signals":len(signals),"strategies":{},
  "limitations":[
   "Three independent entry modes and three exit modes; Seed A/B remain unchanged.",
   "Signal features use completed signal-day OHLC and next-session open entry.",
   "MA200 distance -3% to -15%, rising MA50, 10-session signal cooldown are exploratory.",
   "STOP6/STOP7 use fixed initial stops and 120-session maximum hold; no profit target.",
   "STAGED7 uses initial -7% stop, +8% closing gain trigger, then ATR3 and breakeven floor.",
   "Entry-session stop touch uses daily OHLC and modeled 20bps stress; gaps can exceed stop.",
   "Terminal mark-to-market is synthetic, not a realized exit.",
   "Retrospective universe and repeated 2022-26 research invalidate independent OOS claims.",
   "No live trading, alerts, new capital allocation or automatic promotion.",
   "Same-day entry/stop exits use next-open entry and daily low with 20bps modeled stress.",
   "All nine strategies must validate; otherwise the workflow fails hard."]}
 ledgers=[];curves=[]
 for entry_mode in MODES:
  for exit_mode in EXITS:
   z=candidates[(candidates["mode"]==entry_mode)&(candidates.exit_mode==exit_mode)]
   name=entry_mode+"_"+exit_mode
   if z.empty:
    report["strategies"][name]={"status":"NO_SIGNALS"};continue
   records=[{"ticker":r.ticker,"entry_date":r.entry_date,"exit_date":r.exit_date,
     "entry_price":r.entry_price,"exit_price":r.exit_price,"edge":r.edge,"reason":r.reason}
     for r in z.itertuples(index=False)]
   try:perf,ledger,curve=portfolio_same_day(records,prices,name)
   except Exception as e:
    report["strategies"][name]={"status":"INVALID","error":str(e)};continue
   report["strategies"][name]={**perf,"candidates":len(z)}
   for r in ledger:r["strategy"]=name
   for r in curve:r["strategy"]=name
   ledgers+=ledger;curves+=curve
 if any("status" in x and x["status"]=="INVALID" for x in report["strategies"].values()):
  (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
  raise RuntimeError("INVALID strategy: recovery comparison cannot be accepted")
 if len(report["strategies"])!=9 or any("return" not in x for x in report["strategies"].values()):
  raise RuntimeError("Incomplete nine-strategy frontier")
 pd.DataFrame(ledgers).to_csv(out/"ledger.csv",index=False)
 pd.DataFrame(curves).to_csv(out/"daily_equity.csv",index=False)
 (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
 trs="".join("<tr><td>"+html.escape(k)+"</td><td>"+(f"{v['return']:.2%}" if "return" in v else v["status"])+"</td><td>"+(f"{v['daily_close_mdd']:.2%}" if "daily_close_mdd" in v else "—")+"</td><td>"+str(v.get("trades","—"))+"</td></tr>" for k,v in report["strategies"].items())
 (out/"report.html").write_text("<!doctype html><meta charset='utf-8'><title>PRISM v4.62</title><style>body{background:#0d1829;color:#e8f2ff;font:15px system-ui;margin:28px}h1{color:#61e0b8}table{border-collapse:collapse;width:100%}td,th{padding:11px;border-bottom:1px solid #345}</style><h1>PRISM v4.62 · Recovery 50/200</h1><p>Independent virtual challenger, Seed A/B unchanged. Research only.</p><table><tr><th>Variant</th><th>Return</th><th>Daily MDD</th><th>Trades</th></tr>"+trs+"</table>",encoding="utf-8")
 print(json.dumps({"status":"generated","signals":len(signals),"variants":len(report["strategies"])}))
if __name__=="__main__":main()
