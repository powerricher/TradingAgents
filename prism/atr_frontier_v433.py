"""PRISM v4.33: monotone ATR3/4/5 exit frontier and concentration audit.
Historical research only. Terminal marks are not realized sales.
"""
import argparse,json,html
from pathlib import Path
import numpy as np,pandas as pd
from .trend_v48 import load_prices,portfolio,STOP_SLIPPAGE
from .trend_v47 import atr
MODES=("MA30","ATR3","ATR4","ATR5")
CUTOFF=pd.Timestamp("2026-10-06")
def exit_atr(d,entry,k):
 e=d.index.get_loc(entry)
 av=atr(d);peak=float(d.Open.iloc[e]);stop=-np.inf
 for j in range(e,len(d)):
  op=float(d.Open.iloc[j]);low=float(d.Low.iloc[j]);close=float(d.Close.iloc[j])
  if j>e and np.isfinite(stop):
   if op<=stop:return d.index[j],op,"gap_stop"
   if low<=stop:return d.index[j],stop*(1-STOP_SLIPPAGE),"stop_touch_stress"
  if j-e>=180 and j+1<len(d):
   return d.index[j+1],float(d.Open.iloc[j+1]),"max_hold"
  peak=max(peak,close)
  if pd.notna(av.iloc[j]):
   stop=max(stop,peak-k*float(av.iloc[j]))
 return d.index[-1],float(d.Close.iloc[-1]),"terminal_mark_to_market"
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--scored",default="results/prism-v420/scored_signals.csv");ap.add_argument("--out",default="results/prism-v433");a=ap.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 df=pd.read_csv(a.scored)
 for col in ("date","entry_date","exit_date"):df[col]=pd.to_datetime(df[col]).dt.normalize()
 if df.empty or df.rank_VCP_ALL.isna().any():raise RuntimeError("Missing VCP input")
 prices={t:load_prices(t,g.date.min()).loc[lambda x:x.index<=CUTOFF] for t,g in df.groupby("ticker")}
 for r in df.itertuples(index=False):
  if r.entry_date not in prices[r.ticker].index:raise RuntimeError("Missing entry "+r.ticker)
 result={"version":"v4.33","status":"HISTORICAL / NOT INDEPENDENT OOS","cutoff":str(CUTOFF.date()),"variants":{},"notes":[
 "VCP80 frozen signals and ranks; same 10M KRW initial capital, one-slot, 10bps cost.",
 "ATR stop is monotone non-decreasing and uses prior close information only; no entry-session stop execution.",
 "Stop touch is modeled at stop minus 20bps; gap fills at observed open.",
 "Terminal mark is hypothetical liquidation at cutoff close, not a realized trade.",
 "The existing portfolio() engine books terminal marks as sales; realized and unrealized must be separated in audit.",
 "Repeated historical research and retrospectively selected universe invalidate independent OOS claims.",
 "Daily close MDD excludes intraday troughs."]}
 allledger=[];allcurves=[];alltrades=[]
 for mode in MODES:
  rows=[]
  for r in df.itertuples(index=False):
   d=prices[r.ticker]
   if mode=="MA30":xd,px,reason=r.exit_date,float(r.exit_price),r.reason
   else:xd,px,reason=exit_atr(d,r.entry_date,int(mode[-1]))
   if xd<r.entry_date:raise RuntimeError("Invalid exit chronology")
   rows.append({"ticker":r.ticker,"mode":mode,"entry_date":str(r.entry_date.date()),"exit_date":str(xd.date()),"entry_price":float(r.entry_price),"exit_price":float(px),"edge":float(r.rank_VCP_ALL),"reason":reason})
  perf,ledger,curve=portfolio(rows,prices,mode,rotation=False)
  if perf["trades"]==0:raise RuntimeError("Zero-trade simulation")
  lookup={(x["ticker"],x["entry_date"]):x for x in rows}
  realized=[];terminal=[];pnl=[];all_trade=[]
  prior=10_000_000.
  for z in ledger:
   key=(z["ticker"],z["entry_date"]);r=lookup[key]
   delta=float(z["capital_after"])-prior
   (terminal if r["reason"]=="terminal_mark_to_market" else realized).append(delta)
   pnl.append(delta);prior=float(z["capital_after"])
   all_trade.append({"variant":mode,**z,"pnl_krw":delta,"terminal_mark":r["reason"]=="terminal_mark_to_market"})
  wins=sorted((x for x in pnl if x>0),reverse=True)
  gross=sum(wins)
  result["variants"][mode]={**perf,"terminal_executed_count":len(terminal),
   "realized_pnl_krw":float(sum(realized)),"terminal_mark_pnl_krw":float(sum(terminal)),
   "top1_gross_win_share":float(sum(wins[:1])/gross) if gross else None,
   "top3_gross_win_share":float(sum(wins[:3])/gross) if gross else None,
   "top5_gross_win_share":float(sum(wins[:5])/gross) if gross else None,
   "candidate_terminal_marks":sum(x["reason"]=="terminal_mark_to_market" for x in rows),
   "skipped_signals":perf["skipped"]}
  alltrades+=all_trade
  for x in ledger:x["variant"]=mode
  for x in curve:x["variant"]=mode
  allledger+=ledger;allcurves+=curve
 pd.DataFrame(alltrades).to_csv(out/"trade_attribution.csv",index=False)
 pd.DataFrame(allledger).to_csv(out/"ledger.csv",index=False)
 pd.DataFrame(allcurves).to_csv(out/"daily_equity.csv",index=False)
 (out/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
 lines="".join(f"<tr><td>{html.escape(k)}</td><td>{v['return']:.2%}</td><td>{v['daily_close_mdd']:.2%}</td><td>{v['trades']}</td><td>{v['terminal_executed_count']}</td></tr>" for k,v in result["variants"].items())
 (out/"report.html").write_text("<!doctype html><meta charset='utf-8'><title>PRISM v4.33</title><style>body{background:#0b1729;color:#edf7ff;font:15px system-ui;margin:28px}h1{color:#5de4bd}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #345}</style><h1>PRISM v4.33 · Monotone ATR Exit Frontier</h1><p>Terminal valuation is NOT realized P&amp;L; historical research only.</p><table><tr><th>Strategy</th><th>Compound return</th><th>Daily MDD</th><th>Trades</th><th>Terminal marks</th></tr>"+lines+"</table>",encoding="utf-8")
 print(json.dumps({"status":"generated","variants":{k:v["return"] for k,v in result["variants"].items()}},ensure_ascii=False))
if __name__=="__main__":main()
