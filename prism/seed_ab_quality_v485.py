"""PRISM v4.85: frozen A/B contribution and fragility diagnostics.
No strategy retuning, no in-sample winner promotion, NOT OOS.
"""
import json,html
from pathlib import Path
import numpy as np,pandas as pd
START=10_000_000.
def main():
 out=Path("results/prism-v485");out.mkdir(parents=True,exist_ok=True)
 a=pd.read_csv("results/prism-v436/ledger.csv")
 b=pd.read_csv("results/prism-v457/trade_ledger.csv")
 a=a[a.variant=="ATR4_CONTROL"].copy()
 b=b[b.experiment=="TECH150_VCP_BREAKOUT_QUALITY_RANK"].copy()
 curves={}
 for seed,path,key,value in (
  ("A","results/prism-v436/daily_equity.csv","variant","ATR4_CONTROL"),
  ("B","results/prism-v457/seed_b_equity.csv","experiment","TECH150_VCP_BREAKOUT_QUALITY_RANK")):
  f=pd.read_csv(path);f=f[f[key]==value].copy()
  if f.empty:raise RuntimeError("Missing equity "+seed)
  f["date"]=pd.to_datetime(f.date)
  if f.date.duplicated().any():raise RuntimeError("Duplicate equity date")
  curves[seed]=f.sort_values("date").set_index("date").equity.astype(float)
 rows=[];trades=[]
 for seed,ledger in (("A",a),("B",b)):
  if ledger.empty:raise RuntimeError("Missing ledger "+seed)
  ledger=ledger.sort_values(["exit_date","entry_date","ticker"]).copy()
  ledger["entry_date"]=pd.to_datetime(ledger.entry_date)
  ledger["exit_date"]=pd.to_datetime(ledger.exit_date)
  ledger["prior_capital"]=ledger.capital_after.shift().fillna(START)
  ledger["pnl"]=ledger.capital_after-ledger.prior_capital
  ledger["return_on_prior_capital"]=ledger.capital_after/ledger.prior_capital-1
  ledger["seed"]=seed
  trades.append(ledger[["seed","ticker","entry_date","exit_date","pnl","return_on_prior_capital"]])
  gains=ledger.loc[ledger.pnl>0,"pnl"].sort_values(ascending=False)
  gross=float(gains.sum())
  eq=curves[seed].to_numpy();peak=np.maximum.accumulate(np.r_[START,eq])[1:]
  rows.append({"seed":seed,"trades":len(ledger),
    "return":float(eq[-1]/START-1),"mdd":float(np.min(eq/peak-1)),
    "win_rate":float((ledger.pnl>0).mean()),
    "median_trade_return":float(ledger.return_on_prior_capital.median()),
    "top1_gross_profit_share":float(gains.iloc[:1].sum()/gross) if gross else None,
    "top3_gross_profit_share":float(gains.iloc[:3].sum()/gross) if gross else None,
    "top10_gross_profit_share":float(gains.iloc[:10].sum()/gross) if gross else None})
 pd.DataFrame(rows).to_csv(out/"seed_quality.csv",index=False)
 pd.concat(trades).to_csv(out/"trade_attribution.csv",index=False)
 cal=curves["A"].index.union(curves["B"].index).sort_values()
 aligned={}
 for seed,s in curves.items():
  v=s.reindex(cal).ffill();v.loc[cal<s.index.min()]=START
  if v.isna().any():raise RuntimeError("Equity gap "+seed)
  aligned[seed]=v
 frame=pd.DataFrame(aligned)
 frame["AB"]=frame.A+frame.B
 frame.to_csv(out/"daily_equity.csv",index_label="date")
 yearly=[]
 for year,g in frame.groupby(frame.index.year):
  prior=frame.loc[frame.index<g.index.min()]
  for seed in ("A","B","AB"):
   initial=float(prior[seed].iloc[-1]) if len(prior) else START*(2 if seed=="AB" else 1)
   yearly.append({"year":int(year),"seed":seed,"return":float(g[seed].iloc[-1]/initial-1)})
 pd.DataFrame(yearly).to_csv(out/"yearly_returns.csv",index=False)
 report={"version":"v4.85","status":"FROZEN_A_B_QUALITY_AUDIT_NOT_OOS",
  "quality":rows,"yearly":yearly,
  "limitations":["No new strategies or 2500-name A/B expansion in this version.",
  "A/B trade P&L reconstructed from sequential capital_after; synthetic terminal marks possible.",
  "A and B retain independent capital, not a shared-cash execution engine.",
  "Historical repeated research, present-day universe and missing delistings bias outcomes.",
  "Top trade concentration describes fragility, not a forward predictive signal.",
  "No live orders or strategy promotion; annual +200% remains unverified research target."]}
 (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
 trs="".join("<tr><td>%s</td><td>%.2f%%</td><td>%.2f%%</td><td>%d</td><td>%.1f%%</td></tr>"%(x["seed"],100*x["return"],100*x["mdd"],x["trades"],100*x["top3_gross_profit_share"] if x["top3_gross_profit_share"] is not None else 0) for x in rows)
 (out/"report.html").write_text("<!doctype html><html lang='ko'><meta charset='utf-8'><title>PRISM v4.85</title><style>body{background:#0b1729;color:#eaf4ff;font:15px system-ui;margin:30px}h1{color:#5de4bd}td,th{padding:10px;border-bottom:1px solid #345}</style><h1>PRISM v4.85 A/B Quality</h1><p>Historical frozen audit, NOT OOS.</p><table><tr><th>Seed</th><th>Return</th><th>MDD</th><th>Trades</th><th>Top 3 gross profit share</th></tr>"+trs+"</table></html>",encoding="utf-8")
 print(json.dumps({"status":"V485_COMPLETE","seeds":len(rows)}))
if __name__=="__main__":main()
