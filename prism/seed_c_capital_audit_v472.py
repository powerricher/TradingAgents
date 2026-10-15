"""PRISM v4.72: C1 signal density, idle capital and candidate ranking audit.
Consumes v4.67/v4.68/v4.71 frozen outputs. No rule changes or trade simulation.
"""
import json,html
from pathlib import Path
import numpy as np,pandas as pd
STRATEGY="FOUR_25"
def main():
 out=Path("results/prism-v472");out.mkdir(parents=True,exist_ok=True)
 raw=pd.read_csv("results/prism-v467/signal_forward_returns.csv")
 cand=pd.read_csv("results/prism-v468/candidate_exit_scenarios.csv")
 ledger=pd.read_csv("results/prism-v471/ledger.csv")
 eq=pd.read_csv("results/prism-v471/daily_equity.csv")
 summary471=json.loads(Path("results/prism-v471/summary.json").read_text())
 c=cand[(cand.detector=="C1_VOL_EXPANSION")&(cand.exit_rule=="TP8_SL4")&(cand.hold==5)].copy()
 s=raw[raw.detector=="C1_VOL_EXPANSION"].copy()
 l=ledger[ledger.strategy==STRATEGY].copy()
 e=eq[eq.strategy==STRATEGY].copy()
 if any(x.empty for x in (c,s,l,e)):raise RuntimeError("Missing frozen C1 source data")
 if c.duplicated(["ticker","entry_date"]).any() or s.duplicated(["ticker","entry_date"]).any():
  raise RuntimeError("Duplicate frozen candidate identities")
 if len(c)!=len(s):raise RuntimeError("C1 candidate count mismatch")
 e["date"]=pd.to_datetime(e.date).dt.normalize()
 e=e.sort_values("date")
 if e.date.duplicated().any():raise RuntimeError("Duplicate portfolio dates")
 if not np.all(np.isfinite(e[["equity","cash","positions"]].to_numpy(dtype=float))):
  raise RuntimeError("Nonfinite daily equity or cash")
 if (e.equity<=0).any() or (e.cash<-.01).any():raise RuntimeError("Invalid account balances")
 e["invested_fraction"]=1-e.cash/e.equity
 if (e.invested_fraction<-.00001).any() or (e.invested_fraction>1.00001).any():
  raise RuntimeError("Invalid invested fraction")
 e["week"]=e.date.dt.to_period("W-FRI").astype(str)
 weekly=e.groupby("week",sort=True).agg(last_date=("date","max"),end_equity=("equity","last")).reset_index()
 weekly["weekly_return"]=weekly.end_equity.pct_change()
 weekly.loc[0,"weekly_return"]=weekly.loc[0,"end_equity"]/10_000_000.-1
 weekly.to_csv(out/"weekly_equity_returns.csv",index=False)
 c["entry_date"]=pd.to_datetime(c.entry_date).dt.normalize()
 s["entry_date"]=pd.to_datetime(s.entry_date).dt.normalize()
 c["signal_date"]=pd.to_datetime(c.signal_date).dt.normalize()
 c["gross_win"]=c.exit_price>c.entry_price
 selected=set(zip(l.ticker,pd.to_datetime(l.entry_date).dt.normalize()))
 c["executed"]= [(t,d) in selected for t,d in zip(c.ticker,c.entry_date)]
 if int(c.executed.sum())!=len(l):raise RuntimeError("Executed trade join mismatch")
 c["candidate_rank"]=c.groupby("entry_date")["edge"].rank(method="first",ascending=False)
 c["signal_month"]=c.signal_date.dt.to_period("M").astype(str)
 c.to_csv(out/"candidate_ranking_audit.csv",index=False)
 density=c.groupby("entry_date").agg(signals=("ticker","size"),executed=("executed","sum"),
  unselected=("executed",lambda x:int((~x).sum()))).reset_index()
 density.to_csv(out/"daily_signal_density.csv",index=False)
 monthly=c.groupby("signal_month").agg(signals=("ticker","size"),
  distinct_tickers=("ticker","nunique")).reset_index()
 monthly.to_csv(out/"monthly_signal_density.csv",index=False)
 daily=e.merge(density,left_on="date",right_on="entry_date",how="left")
 daily[["signals","executed","unselected"]]=daily[["signals","executed","unselected"]].fillna(0).astype(int)
 daily.to_csv(out/"capital_utilization.csv",index=False)
 # Descriptive score bins only; no ex-post selection or fitted threshold.
 c["rvol_quantile"]=pd.qcut(c.edge.rank(method="first"),4,labels=["Q1","Q2","Q3","Q4"])
 bins=c.groupby("rvol_quantile",observed=True).agg(candidates=("ticker","size"),
  executed=("executed","sum"),gross_win_rate=("gross_win","mean"),
  mean_gross_return=("gross_win",lambda x:float("nan"))).reset_index()
 means=c.groupby("rvol_quantile",observed=True).apply(
  lambda x:float((x.exit_price/x.entry_price-1).mean()),include_groups=False)
 bins["mean_gross_return"]=bins.rvol_quantile.map(means).astype(float)
 bins.to_csv(out/"rvol_candidate_quartiles.csv",index=False)
 report={"version":"v4.72","status":"FROZEN_SIGNAL_AND_CAPITAL_AUDIT_NOT_OOS",
  "reference_strategy":STRATEGY,"candidates":len(c),"executed":len(l),
  "signal_days":int((daily.signals>0).sum()),"calendar_trading_days":len(daily),
  "no_signal_days":int((daily.signals==0).sum()),
  "mean_invested_fraction":float(e.invested_fraction.mean()),
  "median_invested_fraction":float(e.invested_fraction.median()),
  "mean_cash_fraction":float((e.cash/e.equity).mean()),
  "weeks":len(weekly),"weeks_positive":int((weekly.weekly_return>0).sum()),
  "weeks_negative":int((weekly.weekly_return<0).sum()),
  "weeks_at_least_20pct":int((weekly.weekly_return>=.20).sum()),
  "best_week":float(weekly.weekly_return.max()),
  "worst_week":float(weekly.weekly_return.min()),
  "multi_signal_days":int((density.signals>1).sum()),
  "unselected_signals":int((~c.executed).sum()),
  "reference_return":float(e.equity.iloc[-1]/10_000_000-1),
  "limitations":[
   "Frozen C1 signals and FOUR_25 execution from v4.71; no new portfolio rules.",
   "Unselected candidate gross outcomes are hypothetical independent paths, not executable replacement trades.",
   "RVOL quartiles use all historical candidates and are descriptive, not calibrated predictive scores.",
   "Weekly returns use Friday-ending weeks and last available daily equity, including partial weeks.",
   "Market dates come from frozen equity; no independently verified market calendar.",
   "Current-listed universe and repeated historical research are not point-in-time or independent OOS.",
   "Daily-close MDD misses intraday troughs; no automatic orders or notifications."]}
 expected=summary471["strategies"][STRATEGY]
 if abs(report["reference_return"]-float(expected["return"]))>1e-7:
  raise RuntimeError("v4.71 reference return mismatch")
 (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
 lines="".join(f"<tr><td>{html.escape(k)}</td><td>{html.escape(str(v))}</td></tr>" for k,v in report.items() if k!="limitations")
 (out/"report.html").write_text("<!doctype html><html lang='ko'><meta charset='utf-8'><title>PRISM v4.72</title><style>body{background:#0b1729;color:#eaf4ff;font:15px system-ui;margin:28px}h1{color:#5de4bd}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #345;text-align:left}</style><h1>PRISM v4.72 · Seed C Signal Density &amp; Capital Utilization</h1><p>Frozen C1 FOUR_25. Historical diagnostics, not OOS.</p><table>"+lines+"</table></html>",encoding="utf-8")
 print(json.dumps({"status":"AUDIT_GENERATED","candidate_count":len(c),"signal_days":report["signal_days"]}))
if __name__=="__main__":main()
