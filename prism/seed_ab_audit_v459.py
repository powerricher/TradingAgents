"""PRISM v4.59: independent Seed A/B audit, concentration, overlap, cash time.
Historical diagnostics only; does not change trading strategies.
"""
import json,html
from pathlib import Path
import numpy as np,pandas as pd
from .trend_v48 import load_prices,portfolio
from .atr_frontier_v433 import CUTOFF
START=10_000_000.
def summarize(label,ledger,curve,candidates):
 z=ledger.sort_values(["entry_date","exit_date"]).copy()
 z["prior_capital"]=z.capital_after.shift().fillna(START)
 z["pnl"]=z.capital_after-z.prior_capital
 gains=z[z.pnl>0].sort_values("pnl",ascending=False)
 gross=float(gains.pnl.sum())
 z["entry_ts"]=pd.to_datetime(z.entry_date)
 z["exit_ts"]=pd.to_datetime(z.exit_date)
 z["hold_calendar_days"]=(z.exit_ts-z.entry_ts).dt.days
 terminal=candidates[["ticker","entry_date","reason"]].drop_duplicates(["ticker","entry_date"])
 z=z.merge(terminal,on=["ticker","entry_date"],how="left",suffixes=("","_source"))
 z["terminal_mark"]=z.reason_source.eq("terminal_mark_to_market") if "reason_source" in z else z.reason.eq("terminal_mark_to_market")
 return z,{"trades":len(z),"win_rate":float((z.pnl>0).mean()) if len(z) else None,
  "mean_pnl":float(z.pnl.mean()) if len(z) else None,
  "median_holding_days":float(z.hold_calendar_days.median()) if len(z) else None,
  "top1_gross_profit_share":float(gains.pnl.head(1).sum()/gross) if gross else None,
  "top3_gross_profit_share":float(gains.pnl.head(3).sum()/gross) if gross else None,
  "terminal_marks":int(z.terminal_mark.sum()),
  "end_equity":float(curve.equity.iloc[-1])}
def main():
 out=Path("results/prism-v459");out.mkdir(parents=True,exist_ok=True)
 a=pd.read_csv("results/prism-v436/ledger.csv")
 a=a[a.variant=="ATR4_CONTROL"].copy()
 ac=pd.read_csv("results/prism-v436/daily_equity.csv")
 ac=ac[ac.variant=="ATR4_CONTROL"].copy()
 ar=pd.read_csv("results/prism-v436/candidate_exits.csv")
 ar=ar[ar.variant=="ATR4_CONTROL"].copy()
 b=pd.read_csv("results/prism-v457/trade_ledger.csv")
 b=b[b.experiment=="TECH150_VCP_BREAKOUT_QUALITY_RANK"].copy()
 bc=pd.read_csv("results/prism-v457/seed_b_equity.csv")
 bc=bc[bc.experiment=="TECH150_VCP_BREAKOUT_QUALITY_RANK"].copy()
 br=pd.read_csv("results/prism-v445/candidate_exits.csv")
 br=br[br.detector=="VCP_BREAKOUT"].copy()
 if any(x.empty for x in (a,ac,ar,b,bc,br)):raise RuntimeError("Missing frozen Seed A/B baseline")
 for frame in (ac,bc):
  frame["date"]=pd.to_datetime(frame.date).dt.normalize()
  frame.sort_values("date",inplace=True)
 za,sa=summarize("A",a,ac,ar)
 zb,sb=summarize("B",b,bc,br)
 za["seed"]="A";zb["seed"]="B"
 pd.concat([za,zb],ignore_index=True).to_csv(out/"trade_audit.csv",index=False)
 overlap=[]
 for x in za.itertuples(index=False):
  for y in zb.itertuples(index=False):
   first=max(x.entry_ts,y.entry_ts);last=min(x.exit_ts,y.exit_ts)
   if first<=last:
    overlap.append({"a_ticker":x.ticker,"b_ticker":y.ticker,"from":str(first.date()),
     "to":str(last.date()),"calendar_days":int((last-first).days+1),
     "same_ticker":x.ticker==y.ticker})
 pd.DataFrame(overlap,columns=["a_ticker","b_ticker","from","to","calendar_days","same_ticker"]).to_csv(out/"overlap.csv",index=False)
 ca=ac.set_index("date").equity
 cb=bc.set_index("date").equity
 cal=ca.index.union(cb.index).sort_values()
 xa=ca.reindex(cal).ffill();xb=cb.reindex(cal).ffill()
 xa.loc[cal<ca.index.min()]=START;xb.loc[cal<cb.index.min()]=START
 if xa.isna().any() or xb.isna().any():raise RuntimeError("Equity date gaps")
 combined=xa+xb
 combined.to_frame("combined_equity").assign(seed_a=xa,seed_b=xb).to_csv(out/"combined_equity.csv")
 # Calendar-day occupation proxy, not trading-session cash fraction.
 begin=cal.min();end=cal.max()
 days=pd.date_range(begin,end,freq="D")
 utilization={}
 for label,z in (("A",za),("B",zb)):
  occupied=np.zeros(len(days),dtype=bool)
  for r in z.itertuples(index=False):
   occupied|=(days>=r.entry_ts)&(days<=r.exit_ts)
  utilization[label]={"occupied_calendar_days":int(occupied.sum()),"calendar_days":len(days),
   "occupied_fraction":float(occupied.mean())}
 report={"version":"v4.59","status":"FROZEN BASELINE AUDIT / NOT OOS",
  "seed_a":sa,"seed_b":sb,"overlap_intervals":len(overlap),
  "same_ticker_overlaps":sum(r["same_ticker"] for r in overlap),
  "utilization_calendar_proxy":utilization,
  "combined_return":float(combined.iloc[-1]/(2*START)-1),
  "combined_daily_mdd":float((combined/np.maximum.accumulate(np.r_[2*START,combined.to_numpy()])[1:]-1).min()),
  "limitations":[
   "Seed A/B historical ledgers reused, not independent price-history replay.",
   "Opportunity cost of blocked signals is NOT yet modeled as executable replacement trades.",
   "Cash utilization uses calendar days, not trading sessions; mark-to-market terminal sales are synthetic.",
   "Trade PnL derived from sequential capital_after, not broker-confirmed realized PnL.",
   "Candidate coverage, survivorship and repeated in-sample tuning bias remain.",
   "No new strategies promoted, no alerts or live orders."]}
 (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
 (out/"report.html").write_text("<!doctype html><meta charset='utf-8'><title>PRISM v4.59</title><style>body{background:#0c182a;color:#e9f3ff;font:15px system-ui;margin:28px}h1{color:#65e6bf}pre{white-space:pre-wrap}</style><h1>PRISM v4.59 · Seed A/B Audit</h1><p>Frozen historical strategy comparison. No strategy changes.</p><pre>"+html.escape(json.dumps(report,ensure_ascii=False,indent=2))+"</pre>",encoding="utf-8")
 print(json.dumps({"status":"generated","a_trades":len(za),"b_trades":len(zb)}))
if __name__=="__main__":main()
