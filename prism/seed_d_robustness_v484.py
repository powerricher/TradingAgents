"""PRISM v4.84 D4 robustness audit: pre-existing 18 policies, no retuning.
Historical descriptive diagnostics, NOT OOS.
"""
import json,html
from pathlib import Path
import numpy as np,pandas as pd
def main():
 out=Path("results/prism-v484");out.mkdir(parents=True,exist_ok=True)
 src=Path("results/prism-v482")
 summary=json.loads((src/"summary.json").read_text())
 curves=pd.read_csv(src/"daily_equity.csv")
 ledger=pd.read_csv(src/"ledger.csv")
 curves["date"]=pd.to_datetime(curves.date)
 ledger["entry_date"]=pd.to_datetime(ledger.entry_date)
 rows=[];yearly=[]
 for name,g in curves.groupby("strategy"):
  g=g.sort_values("date")
  if g.date.duplicated().any():raise RuntimeError("Duplicate dates "+name)
  e=g.equity.astype(float).to_numpy()
  if not np.isfinite(e).all() or (e<=0).any():raise RuntimeError("Invalid equity "+name)
  if abs(e[-1]-summary["strategies"][name]["final_krw"])>max(1,e[-1]*1e-6):
   raise RuntimeError("v4.82 reconciliation failed "+name)
  trade=ledger[ledger.strategy==name].sort_values("entry_date")
  by_year=[]
  for year,z in g.groupby(g.date.dt.year):
   prior=g[g.date<z.date.min()]
   initial=float(prior.equity.iloc[-1]) if len(prior) else 10_000_000.
   ret=float(z.equity.iloc[-1]/initial-1)
   yearly.append({"strategy":name,"year":int(year),"return":ret})
   by_year.append(ret)
  daily=g.equity.pct_change().dropna()
  rows.append({"strategy":name,"overall_return":float(e[-1]/10_000_000-1),
    "mdd":float(np.min(e/np.maximum.accumulate(np.r_[10_000_000,e])[1:]-1)),
    "trades":len(trade),"positive_years":int(sum(x>0 for x in by_year)),
    "negative_years":int(sum(x<0 for x in by_year)),
    "worst_year_return":float(min(by_year)),
    "median_year_return":float(np.median(by_year)),
    "worst_daily_return":float(daily.min()) if len(daily) else None,
    "terminal_open_position":bool(summary["strategies"][name]["terminal_open_position"])})
 pd.DataFrame(rows).to_csv(out/"strategy_robustness.csv",index=False)
 pd.DataFrame(yearly).to_csv(out/"yearly_returns.csv",index=False)
 report={"version":"v4.84","status":"FROZEN_D4_ROBUSTNESS_AUDIT_NOT_OOS",
 "strategies":rows,"limitations":[
  "18 already-tested D4 configurations, no new tuning or strategy promotion.",
  "Historical 2022-2026 results; 2026 is partial, not full-year.",
  "Yearly returns and daily MDD based on marked-to-market equity; intraday risk understated.",
  "Prior v4.82 stop-first OHLC, ETF daily reset and cost assumptions persist.",
  "Selection among 18 variants on same data is in-sample and not independent OOS.",
  "No ABCD shared-cash portfolio or execution guarantee."]}
 (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
 trs="".join("<tr><td>%s</td><td>%.2f%%</td><td>%.2f%%</td><td>%d</td><td>%.2f%%</td></tr>"%(html.escape(x["strategy"]),100*x["overall_return"],100*x["mdd"],x["positive_years"],100*x["worst_year_return"]) for x in rows)
 (out/"report.html").write_text("<!doctype html><html lang='ko'><meta charset='utf-8'><title>PRISM v4.84</title><style>body{background:#0b1729;color:#eaf4ff;font:15px system-ui;margin:30px}h1{color:#5de4bd}td,th{padding:10px;border-bottom:1px solid #345}</style><h1>PRISM v4.84 · D4 robustness</h1><p>Historical audit, not OOS. No C4 tuning.</p><table><tr><th>Strategy</th><th>Return</th><th>MDD</th><th>Positive years</th><th>Worst year</th></tr>"+trs+"</table></html>",encoding="utf-8")
 print(json.dumps({"status":"V484_COMPLETE","strategies":len(rows)}))
if __name__=="__main__":main()
