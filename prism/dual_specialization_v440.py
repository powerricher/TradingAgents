"""PRISM v4.40 independent seed role audit.
Seed A ATR4 trend capture, Seed B MA30 opportunity rotation.
Reuses v4.36 executions and v4.39 equity; no new model fitted.
"""
import argparse,json,html
from pathlib import Path
import numpy as np,pandas as pd
SEED=10_000_000.
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--equity",default="results/prism-v439/dual_daily_equity.csv")
 p.add_argument("--ledger",default="results/prism-v436/ledger.csv")
 p.add_argument("--candidates",default="results/prism-v436/candidate_exits.csv")
 p.add_argument("--out",default="results/prism-v440")
 a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 eq=pd.read_csv(a.equity);eq=eq[eq.pair=="MA30_ATR4"].copy()
 led=pd.read_csv(a.ledger);cand=pd.read_csv(a.candidates)
 if eq.empty:raise RuntimeError("Missing dual seed curve")
 for d in (eq,led,cand):
  for col in ("date","entry_date","exit_date"):
   if col in d:d[col]=pd.to_datetime(d[col]).dt.normalize()
 eq=eq.sort_values("date").reset_index(drop=True)
 if eq.date.duplicated().any():raise RuntimeError("Duplicate trading days")
 legs={"A_TREND_ATR4":"ATR4_CONTROL","B_ROTATION_MA30":"MA30_CONTROL"}
 trades={};positions={};metrics={}
 for role,variant in legs.items():
  x=led[led.variant==variant].sort_values("entry_date").copy()
  c=cand[cand.variant==variant].copy()
  if x.empty or c.empty:raise RuntimeError("Missing execution "+variant)
  x=x.merge(c[["ticker","entry_date","reason"]],on=["ticker","entry_date"],how="left",validate="one_to_one")
  if x.reason.isna().any():raise RuntimeError("Unmatched exits "+variant)
  trades[role]=x
  # Position is open after its entry session open and before its exit session open.
  pos=[]
  for day in eq.date:
   active=x[(x.entry_date<=day)&(x.exit_date>day)]
   if len(active)>1:raise RuntimeError("Overlapping positions within "+variant)
   pos.append(str(active.iloc[0].ticker) if len(active) else "")
  positions[role]=pos
  held=np.array([bool(v) for v in pos])
  metrics[role]={"trades":len(x),"invested_session_fraction":float(held.mean()),
   "cash_session_fraction":float((~held).mean()),
   "mean_holding_calendar_days":float((x.exit_date-x.entry_date).dt.days.mean()),
   "terminal_mark_count":int((x.reason=="terminal_mark_to_market").sum()),
   "distinct_tickers":int(x.ticker.nunique())}
 eq["ticker_a"]=positions["A_TREND_ATR4"];eq["ticker_b"]=positions["B_ROTATION_MA30"]
 eq["both_invested"]=(eq.ticker_a!="")&(eq.ticker_b!="")
 eq["same_ticker"]=eq.both_invested&(eq.ticker_a==eq.ticker_b)
 eq["only_a"]=(eq.ticker_a!="")&(eq.ticker_b=="")
 eq["only_b"]=(eq.ticker_a=="")&(eq.ticker_b!="")
 eq["both_cash"]=(eq.ticker_a=="")&(eq.ticker_b=="")
 eq["ret_a"]=eq.seed_b_equity.pct_change().fillna(eq.seed_b_equity/SEED-1)
 eq["ret_b"]=eq.seed_a_equity.pct_change().fillna(eq.seed_a_equity/SEED-1)
 # v4.39 columns: seed_a=MA30, seed_b=ATR4.
 corr=float(eq.ret_a.corr(eq.ret_b))
 if not np.isfinite(corr):raise RuntimeError("Undefined daily correlation")
 eq["year"]=eq.date.dt.year
 annual={};previous_a=previous_b=SEED
 for year,g in eq.groupby("year"):
  end_a=float(g.seed_b_equity.iloc[-1]);end_b=float(g.seed_a_equity.iloc[-1])
  annual[str(year)]={"atr4":end_a/previous_a-1,"ma30":end_b/previous_b-1}
  previous_a=end_a;previous_b=end_b
 report={"version":"v4.40","status":"HISTORICAL ROLE SPECIALIZATION AUDIT / NOT INDEPENDENT OOS",
  "capital_total":2*SEED,"roles":metrics,
  "overlap":{"both_invested_fraction":float(eq.both_invested.mean()),
   "same_ticker_fraction_all_sessions":float(eq.same_ticker.mean()),
   "same_ticker_fraction_when_both_invested":float(eq.loc[eq.both_invested,"same_ticker"].mean()) if eq.both_invested.any() else None,
   "only_trend_fraction":float(eq.only_a.mean()),"only_rotation_fraction":float(eq.only_b.mean()),
   "both_cash_fraction":float(eq.both_cash.mean()),
   "daily_return_correlation":corr},
  "annual_returns":annual,
  "combined_return":float(eq.total_equity.iloc[-1]/(2*SEED)-1),
  "combined_mdd":float(np.min(eq.total_equity.to_numpy()/np.maximum.accumulate(np.r_[2*SEED,eq.total_equity.to_numpy()])[1:]-1)),
  "limitations":[
   "Role specialization is by existing exit rules only; distinct role-specific entry models are NOT yet trained.",
   "100-stock expansion is NOT included; original VCP80 frozen universe.",
   "No cross-account transfers, leverage, or deduplication of same-ticker holdings.",
   "Invested/cash days use entry/exit date intervals; partial-day exposure not measured.",
   "Daily return correlation includes cash days and terminal mark valuation.",
   "Terminal marks are synthetic valuations; repeated historical model research is not independent OOS.",
   "Equity curve starts before first trading signal; cash fractions include inactive history."]}
 eq.to_csv(out/"daily_role_exposure.csv",index=False)
 pd.concat([x.assign(role=role) for role,x in trades.items()],ignore_index=True).to_csv(out/"role_trade_ledger.csv",index=False)
 (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
 (out/"report.html").write_text("<!doctype html><meta charset='utf-8'><title>PRISM v4.40</title><style>body{background:#0b1729;color:#eaf4ff;font:15px system-ui;margin:28px}h1{color:#5de4bd}pre{white-space:pre-wrap}</style><h1>PRISM v4.40 · Dual-Seed Specialization</h1><p>ATR4 Trend Capture + MA30 Opportunity Rotation · 2 × KRW 10M</p><pre>"+html.escape(json.dumps(report,ensure_ascii=False,indent=2))+"</pre>",encoding="utf-8")
 print(json.dumps({"status":"generated","correlation":corr,"combined_return":report["combined_return"]}))
if __name__=="__main__":main()
