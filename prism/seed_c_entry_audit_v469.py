"""PRISM v4.69 — frozen C1 execution diagnostics, no trading rule changes."""
import json,html
from pathlib import Path
import numpy as np,pandas as pd
NAME="C1_VOL_EXPANSION_TP8_SL4_H5"
START=10_000_000.
def main():
 out=Path("results/prism-v469");out.mkdir(parents=True,exist_ok=True)
 ledger=pd.read_csv("results/prism-v468/ledger.csv")
 candidates=pd.read_csv("results/prism-v468/candidate_exit_scenarios.csv")
 signals=pd.read_csv("results/prism-v467/signal_forward_returns.csv")
 curve=pd.read_csv("results/prism-v468/daily_equity.csv")
 l=ledger[ledger.strategy==NAME].copy()
 c=candidates[(candidates.detector=="C1_VOL_EXPANSION")&(candidates.exit_rule=="TP8_SL4")&(candidates.hold==5)].copy()
 s=signals[signals.detector=="C1_VOL_EXPANSION"].copy()
 e=curve[curve.strategy==NAME].copy()
 if l.empty or c.empty or e.empty:raise RuntimeError("Frozen C1 inputs missing")
 if l.duplicated(["ticker","entry_date"]).any():raise RuntimeError("Duplicate executed trade keys")
 if c.duplicated(["ticker","entry_date"]).any():raise RuntimeError("Duplicate candidate keys")
 if s.duplicated(["ticker","entry_date"]).any():raise RuntimeError("Duplicate original signal keys")
 if len(c)!=len(s):raise RuntimeError("Candidate and signal counts differ")
 l=l.sort_values(["exit_date","entry_date"]).reset_index(drop=True)
 l["pnl_krw"]=l.capital_after.diff().fillna(l.capital_after-START)
 l["entry_year"]=pd.to_datetime(l.entry_date).dt.year
 l["same_day"]=l.entry_date==l.exit_date
 detail=l.merge(c[["ticker","entry_date","exit_date","entry_price","exit_price","gross_return","reason"]],
  on=["ticker","entry_date","exit_date"],how="left",validate="one_to_one",indicator=True)
 if (detail["_merge"]!="both").any():raise RuntimeError("Ledger/candidate mismatch")
 detail=detail.drop(columns="_merge")
 detail=detail.merge(s[["ticker","entry_date","signal_close","signal_rvol","prior_dollar_volume","prior_atr_pct"]],
  on=["ticker","entry_date"],how="left",validate="one_to_one")
 if detail.signal_close.isna().any():raise RuntimeError("Signal feature join failed")
 detail["overnight_gap"]=detail.entry_price/detail.signal_close-1
 detail["loss_trade"]=detail.pnl_krw<0
 detail.to_csv(out/"executed_trade_diagnostics.csv",index=False)
 # Candidate-level gaps are descriptive only, never post-hoc trading filters.
 candidates_join=c.merge(s[["ticker","entry_date","signal_close","signal_rvol","prior_dollar_volume","prior_atr_pct"]],
  on=["ticker","entry_date"],validate="one_to_one")
 candidates_join["overnight_gap"]=candidates_join.entry_price/candidates_join.signal_close-1
 candidates_join["same_day"]=candidates_join.entry_date==candidates_join.exit_date
 candidates_join.to_csv(out/"all_candidate_diagnostics.csv",index=False)
 annual=detail.groupby("entry_year").agg(trades=("ticker","size"),
  pnl_krw=("pnl_krw","sum"),same_day=("same_day","sum"),
  wins=("pnl_krw",lambda x:int((x>0).sum()))).reset_index()
 annual.to_csv(out/"entry_year_attribution.csv",index=False)
 gains=detail[detail.pnl_krw>0].sort_values("pnl_krw",ascending=False)
 total_gain=float(gains.pnl_krw.sum())
 concentration={f"top{n}_share_of_gross_wins":float(gains.head(n).pnl_krw.sum()/total_gain) if total_gain else None for n in (1,3,5)}
 # Ex-post winner exclusion is a concentration diagnostic, NOT a rebalanced portfolio.
 concentration.update({f"top{n}_removed_arithmetic_pnl_krw":float(detail.pnl_krw.sum()-gains.head(n).pnl_krw.sum()) for n in (1,3,5)})
 detail["gap_band"]=pd.cut(detail.overnight_gap,[-np.inf,-.03,0,.03,.06,.10,np.inf],
  labels=["<-3%","-3~0%","0~3%","3~6%","6~10%",">=10%"],right=False)
 gap=detail.groupby("gap_band",observed=True).agg(trades=("ticker","size"),
  same_day=("same_day","sum"),losses=("loss_trade","sum"),
  mean_pnl=("pnl_krw","mean"),median_gap=("overnight_gap","median")).reset_index()
 gap.to_csv(out/"gap_failure_audit.csv",index=False)
 # Signals whose entry falls strictly within an executed position's occupancy.
 # This is a count, not executable replacement P&L.
 blocked=[]
 for r in detail.itertuples(index=False):
  z=candidates_join[(candidates_join.entry_date>r.entry_date)&(candidates_join.entry_date<r.exit_date)]
  blocked.append({"held_ticker":r.ticker,"entry_date":r.entry_date,"exit_date":r.exit_date,
   "blocked_signals":len(z),"positive_gross_candidates":int((z.gross_return>0).sum())})
 pd.DataFrame(blocked).to_csv(out/"blocked_signal_audit.csv",index=False)
 e=e.sort_values("date")
 vals=e.equity.to_numpy(dtype=float)
 if len(vals)<2:raise RuntimeError("Missing daily curve")
 peak=np.maximum.accumulate(np.r_[START,vals])[1:]
 report={"version":"v4.69","status":"FROZEN_C1_DIAGNOSTIC_NOT_NEW_BACKTEST",
  "trades":len(detail),"same_day":int(detail.same_day.sum()),
  "same_day_stops":int((detail.same_day & detail.reason.isin(["stop_touch","gap_stop"])).sum()),
  "win_rate":float((detail.pnl_krw>0).mean()),
  "return":float(vals[-1]/START-1),"daily_close_mdd":float(np.min(vals/peak-1)),
  "net_pnl_krw":float(detail.pnl_krw.sum()),"winner_concentration":concentration,
  "blocked_signal_count":int(sum(x["blocked_signals"] for x in blocked)),
  "limitations":[
   "Historical v4.68 C1 TP8 SL4 H5 signals and fills are frozen; no parameter fitting.",
   "Annual P&L grouped by entry year, not year of actual realization.",
   "Winner removal is arithmetic sensitivity, not counterfactual portfolio replay.",
   "Overnight gap association is observational, not proof of a profitable filter.",
   "Daily OHLC stop-first ordering is a conservative scenario, not intraday execution.",
   "Capital is fully invested in one position, and close-based MDD misses intraday losses.",
   "Retrospective current-listed universe and repeated research are not OOS.",
   "No live trading, automatic alerts or strategy promotion."]}
 if abs(float(detail.pnl_krw.sum())-(float(l.capital_after.iloc[-1])-START))>1:
  raise RuntimeError("Ledger PnL reconciliation failed")
 (out/"summary.json").write_text(json.dumps(report,indent=2,ensure_ascii=False))
 (out/"report.html").write_text("<!doctype html><html lang='ko'><meta charset='utf-8'><title>PRISM v4.69</title><style>body{background:#0b1729;color:#eaf4ff;font:15px system-ui;margin:28px}h1{color:#5de4bd}pre{white-space:pre-wrap}</style><h1>PRISM v4.69 · C1 Entry Failure Audit</h1><p>Frozen C1 +8%/-4%, maximum 5 sessions. Historical diagnostic only.</p><pre>"+html.escape(json.dumps(report,ensure_ascii=False,indent=2))+"</pre>",encoding="utf-8")
 print(json.dumps({"status":"AUDIT_COMPLETE","trades":len(detail),"same_day":int(detail.same_day.sum())}))
if __name__=="__main__":main()
