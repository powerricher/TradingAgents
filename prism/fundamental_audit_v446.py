"""PRISM v4.46: technical signal collision + point-in-time fundamental audit.
No fabricated historical financial data. Fundamentals accepted only from a
dated, externally curated filing snapshot CSV with public availability dates.
"""
import argparse,json,html
from pathlib import Path
import numpy as np,pandas as pd
REQUIRED=("ticker","public_date","revenue_yoy","operating_margin","fcf_margin","net_debt_to_ebitda","valuation_to_growth")
def main():
 p=argparse.ArgumentParser();p.add_argument("--candidates",default="results/prism-v445/candidate_exits.csv")
 p.add_argument("--ledger",default="results/prism-v445/ledger.csv")
 p.add_argument("--fundamentals",default="data/prism/fundamentals_pit.csv")
 p.add_argument("--out",default="results/prism-v446");a=p.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 c=pd.read_csv(a.candidates);ledger=pd.read_csv(a.ledger)
 for col in ("entry_date","exit_date"):
  c[col]=pd.to_datetime(c[col]).dt.normalize()
 c["ticker"]=c.ticker.astype(str)
 # Candidate collision audit: same day competing strategies and symbols.
 day=c.groupby("entry_date").agg(candidate_rows=("ticker","size"),distinct_tickers=("ticker","nunique"),
  detector_count=("detector","nunique")).reset_index()
 collisions=day[day.detector_count>1].copy()
 c["entry_day_detector_count"]=c.entry_date.map(day.set_index("entry_date").detector_count)
 # Detect duplicated exact ticker-entry pairs across detectors.
 dup=c.groupby(["ticker","entry_date"]).detector.nunique().reset_index(name="detectors")
 dup=dup[dup.detectors>1]
 c.to_csv(out/"candidate_audit.csv",index=False)
 collisions.to_csv(out/"signal_collisions.csv",index=False)
 dup.to_csv(out/"same_ticker_dual_signals.csv",index=False)
 # Selected executions under the historical v4.45 alternatives.
 selections=[]
 for x in ledger.itertuples(index=False):
  selections.append({"experiment":x.experiment,"ticker":x.ticker,"entry_date":x.entry_date})
 selected=pd.DataFrame(selections)
 selected.to_csv(out/"executed_selections.csv",index=False)
 report={"version":"v4.46","status":"PIT FUNDAMENTAL COVERAGE AUDIT / NOT A VALIDATED FUNDAMENTAL BACKTEST",
 "candidates":len(c),"collision_days":len(collisions),"same_ticker_dual_signals":len(dup),
 "executed":len(selected),"fundamentals":{},"strategy_promotion":"BLOCKED_UNTIL_PIT_COVERAGE",
 "limitations":[
 "No present-day fundamentals are backfilled into historical trading decisions.",
 "Public_date must be the earliest verified investor-available filing/release date, not fiscal period end.",
 "The fundamental CSV must be independently sourced and audited for restatements, corporate actions and point-in-time values.",
 "Missing values remain missing and cannot be scored as neutral or positive.",
 "This audit does not claim improved P&L or causal fundamental alpha.",
 "v4.45 uses retrospective universe and excludes unresolved exits; no independent OOS.",
 "Cross-detector ranking scores in v4.45 are not calibrated; collision audit does not correct the old portfolio engine."]}
 file=Path(a.fundamentals)
 if file.exists():
  f=pd.read_csv(file)
  missing=set(REQUIRED)-set(f.columns)
  if missing:raise RuntimeError("PIT CSV missing columns "+str(sorted(missing)))
  f.public_date=pd.to_datetime(f.public_date,errors="raise").dt.normalize()
  if f[list(REQUIRED[2:])].apply(pd.to_numeric,errors="coerce").isna().any().any():
   raise RuntimeError("PIT CSV contains missing or nonnumeric required metrics")
  if f.duplicated(["ticker","public_date"]).any():raise RuntimeError("Duplicate ticker public_date")
  f=f.sort_values(["ticker","public_date"])
  matched=pd.merge_asof(c.sort_values("entry_date"),f.sort_values("public_date"),
   left_on="entry_date",right_on="public_date",by="ticker",direction="backward")
  # Conservative publication lag: the same-day filing is not assumed tradable
  # at an unknown intraday release time.
  matched.loc[matched.public_date>=matched.entry_date,list(REQUIRED[2:])]=np.nan
  matched["pit_available"]=matched[list(REQUIRED[2:])].notna().all(axis=1)
  matched.to_csv(out/"fundamental_candidate_coverage.csv",index=False)
  report["fundamentals"]={"status":"USER_PROVIDED_PIT_CSV_UNVERIFIED","rows":len(f),
   "covered_candidates":int(matched.pit_available.sum()),
   "coverage_rate":float(matched.pit_available.mean())}
 else:
  pd.DataFrame(columns=["ticker","public_date",*REQUIRED[2:]]).to_csv(out/"fundamental_input_template.csv",index=False)
  report["fundamentals"]={"status":"MISSING_PIT_SOURCE","covered_candidates":0,
   "coverage_rate":0.0,"required_columns":list(REQUIRED),
   "note":"No historical fundamental rankings or returns were computed."}
 (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str))
 (out/"report.html").write_text("<!doctype html><meta charset='utf-8'><title>PRISM v4.46</title><style>body{background:#0b1729;color:#eaf4ff;font:15px system-ui;margin:28px}h1{color:#5de4bd}pre{white-space:pre-wrap}</style><h1>PRISM v4.46 · Technical × Fundamental Audit</h1><p>Historical fundamental ranking blocked until verified point-in-time snapshots are available.</p><pre>"+html.escape(json.dumps(report,ensure_ascii=False,indent=2,default=str))+"</pre>",encoding="utf-8")
 print(json.dumps({"status":"audit_generated","fundamentals":report["fundamentals"]["status"],"collision_days":len(collisions)}))
if __name__=="__main__":main()
