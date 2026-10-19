"""PRISM v4.75 — fixed C1 BASE vs Base Priority robustness audit.
Historical chronological split is descriptive, NOT untouched OOS.
"""
import json,html
from pathlib import Path
import numpy as np,pandas as pd
from .regime_adaptive_v436 import regimes
MODES=("BASE_RVOL","EXPANDED_BASE_PRIORITY")
def main():
 out=Path("results/prism-v475");out.mkdir(parents=True,exist_ok=True)
 cand=pd.read_csv("results/prism-v474/candidate_quality_audit.csv")
 led=pd.read_csv("results/prism-v474/ledger.csv")
 eq=pd.read_csv("results/prism-v474/daily_equity.csv")
 ref=json.loads(Path("results/prism-v474/summary.json").read_text())
 if any(x.empty for x in (cand,led,eq)):raise RuntimeError("Missing v4.74 reference")
 if not {"ticker","entry_date","exit_date","is_base","quality_score","signal_rvol","gross_return"}.issubset(cand):
  raise RuntimeError("Candidate schema mismatch")
 market=regimes(pd.Timestamp("2022-01-01"))
 cand["entry_date"]=pd.to_datetime(cand.entry_date).dt.normalize()
 cand["regime"]=cand.entry_date.map(market)
 if cand.regime.isna().any() or (cand.regime=="UNKNOWN").any():raise RuntimeError("Missing prior-close QQQ regime")
 cand["period"]=np.where(cand.entry_date<pd.Timestamp("2025-01-01"),"2022_2024","2025_2026")
 cand["positive"]=cand.gross_return>0
 group=cand.groupby(["period","regime","is_base"]).agg(
  candidates=("ticker","size"),positive_rate=("positive","mean"),
  mean_gross_return=("gross_return","mean"),
  median_gross_return=("gross_return","median")).reset_index()
 group.to_csv(out/"candidate_regime_period.csv",index=False)
 # Paired same-day ranking comparison; candidate gross outcomes are hypothetical.
 competition=[]
 for day,g in cand.groupby("entry_date"):
  if len(g)<2:continue
  a=g.sort_values(["signal_rvol","ticker"],ascending=[False,True]).iloc[0]
  b=g.assign(priority=1000*g.is_base.astype(int)+g.quality_score).sort_values(
   ["priority","ticker"],ascending=[False,True]).iloc[0]
  competition.append({"entry_date":str(day.date()),"rvol_ticker":a.ticker,
   "priority_ticker":b.ticker,"changed":a.ticker!=b.ticker,
   "rvol_gross":float(a.gross_return),"priority_gross":float(b.gross_return),
   "priority_minus_rvol_gross":float(b.gross_return-a.gross_return),
   "rvol_is_base":bool(a.is_base),"priority_is_base":bool(b.is_base)})
 pd.DataFrame(competition).to_csv(out/"same_day_counterfactual_candidates.csv",index=False)
 report={"version":"v4.75","status":"FROZEN_STRATEGY_ROBUSTNESS_NOT_OOS",
  "strategies":{},"candidate_regimes":group.to_dict("records"),
  "competition_days":len(competition),
  "ranking_changes":sum(x["changed"] for x in competition),
  "limitations":[
   "Historical time split is descriptive, not independent out-of-sample.",
   "QQQ regime uses prior-close information, not future regime labels.",
   "Candidate gross returns assume independent fills; cannot substitute for portfolio counterfactual P&L.",
   "Per-year trade P&L is attributed by entry year; not realized calendar-year performance.",
   "Equity returns use last available close of each period, including partial periods.",
   "Same-day rank competition is not necessarily an executed trade; occupied capital can block entries.",
   "Repeated research, current-listed universe, and daily OHLC stop-first assumptions remain.",
   "No trading, live alerts, or promotion of a challenger."]}
 rows=[];yearrows=[]
 for name in MODES:
  e=eq[eq.strategy==name].copy()
  l=led[led.strategy==name].copy()
  if e.empty or l.empty:raise RuntimeError("Missing strategy "+name)
  e["date"]=pd.to_datetime(e.date).dt.normalize()
  e=e.sort_values("date")
  l["entry_date"]=pd.to_datetime(l.entry_date).dt.normalize()
  if e.date.duplicated().any():raise RuntimeError("Duplicate equity dates")
  end=float(e.equity.iloc[-1])
  if abs(end-float(ref["strategies"][name]["final_krw"]))>max(1,end*1e-6):
   raise RuntimeError("v4.74 equity mismatch "+name)
  for period,start,stop in (("2022_2024","2022-01-01","2024-12-31"),("2025_2026","2025-01-01","2026-10-06")):
   x=e[(e.date>=start)&(e.date<=stop)]
   if x.empty:raise RuntimeError("Empty period "+period)
   earlier=e[e.date<start]
   initial=float(earlier.equity.iloc[-1]) if not earlier.empty else 10_000_000.
   vals=x.equity.to_numpy(dtype=float)
   peak=np.maximum.accumulate(np.r_[initial,vals])[1:]
   report["strategies"].setdefault(name,{})[period]={
    "return":float(vals[-1]/initial-1),"mdd":float(np.min(vals/peak-1)),
    "start_equity":initial,"end_equity":float(vals[-1])}
  l["year"]=l.entry_date.dt.year
  yr=l.groupby("year").agg(trades=("ticker","size"),pnl_krw=("pnl_krw","sum")).reset_index()
  yr["strategy"]=name
  yearrows.extend(yr.to_dict("records"))
  report["strategies"][name]["trades"]=len(l)
  report["strategies"][name]["overall_return"]=float(end/10_000_000.-1)
  rows.append({"strategy":name,"final_krw":end})
 pd.DataFrame(yearrows).to_csv(out/"yearly_trade_attribution.csv",index=False)
 pd.DataFrame(rows).to_csv(out/"strategy_reconciliation.csv",index=False)
 (out/"summary.json").write_text(json.dumps(report,indent=2,ensure_ascii=False))
 (out/"report.html").write_text("<!doctype html><html lang='ko'><meta charset='utf-8'><title>PRISM v4.75</title><style>body{background:#0b1729;color:#eaf4ff;font:15px system-ui;margin:28px}h1{color:#5de4bd}pre{white-space:pre-wrap}</style><h1>PRISM v4.75 · C1 Robustness Audit</h1><p>Fixed BASE vs Expanded Base Priority, historical split not OOS.</p><pre>"+html.escape(json.dumps(report,indent=2,ensure_ascii=False))+"</pre>",encoding="utf-8")
 print(json.dumps({"status":"AUDIT_GENERATED","ranking_changes":report["ranking_changes"]}))
if __name__=="__main__":main()
