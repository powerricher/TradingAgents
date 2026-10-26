"""PRISM v4.86: frozen Seed A/B trade concentration and loss-side diagnostics.
No selection rule learned from outcomes; not OOS.
"""
import json,html
from pathlib import Path
import numpy as np,pandas as pd
def main():
 out=Path("results/prism-v486");out.mkdir(parents=True,exist_ok=True)
 z=pd.read_csv("results/prism-v485/trade_attribution.csv")
 required={"seed","ticker","entry_date","exit_date","pnl","return_on_prior_capital"}
 if z.empty or not required.issubset(z):raise RuntimeError("Missing v4.85 trade audit")
 z["entry_date"]=pd.to_datetime(z.entry_date)
 z["exit_date"]=pd.to_datetime(z.exit_date)
 z["hold_days"]=(z.exit_date-z.entry_date).dt.days
 z["year"]=z.entry_date.dt.year
 z["positive"]=z.pnl>0
 z["negative"]=z.pnl<0
 rows=[];annual=[];leaveout=[]
 for seed,g in z.groupby("seed"):
  g=g.sort_values("entry_date")
  gains=g[g.pnl>0].sort_values("pnl",ascending=False)
  losses=g[g.pnl<0].sort_values("pnl")
  total=float(g.pnl.sum())
  gross=float(gains.pnl.sum());gross_loss=float(-losses.pnl.sum())
  rows.append({"seed":seed,"trades":len(g),"total_pnl":total,
   "profit_factor":gross/gross_loss if gross_loss>0 else None,
   "median_hold_calendar_days":float(g.hold_days.median()),
   "mean_winner_pnl":float(gains.pnl.mean()) if len(gains) else None,
   "mean_loser_pnl":float(losses.pnl.mean()) if len(losses) else None,
   "median_winner_return":float(gains.return_on_prior_capital.median()) if len(gains) else None,
   "median_loser_return":float(losses.return_on_prior_capital.median()) if len(losses) else None,
   "top3_gross_profit_share":float(gains.pnl.head(3).sum()/gross) if gross else None})
  for k in (1,3,5):
   removed=gains.head(k)
   leaveout.append({"seed":seed,"remove_top_winners":k,"remaining_pnl":float(total-removed.pnl.sum()),
    "remaining_trades":len(g)-len(removed),"remaining_wins":int((g.pnl>0).sum()-len(removed))})
  for year,h in g.groupby("year"):
   annual.append({"seed":seed,"year":int(year),"trades":len(h),
    "win_rate":float(h.positive.mean()),"sum_pnl":float(h.pnl.sum()),
    "median_trade_return":float(h.return_on_prior_capital.median())})
 pd.DataFrame(rows).to_csv(out/"profit_loss_diagnostics.csv",index=False)
 pd.DataFrame(leaveout).to_csv(out/"top_winner_removal.csv",index=False)
 pd.DataFrame(annual).to_csv(out/"yearly_trade_diagnostics.csv",index=False)
 report={"version":"v4.86","status":"FROZEN_A_B_FRAGILITY_DIAGNOSTIC_NOT_OOS",
 "diagnostics":rows,"top_winner_removal":leaveout,
 "limitations":["Removing top winners subtracts their realized PnL, NOT a valid counterfactual compounded portfolio.",
 "Trade PnL comes from sequential capital_after snapshots; synthetic terminal marks may exist.",
 "No new 2500-name Seed A/B replay or learned predictive filter in this version.",
 "Historical sample and repeated strategy selection are not untouched OOS.",
 "No trading, no strategy promotion; annual +200% is unverified research target."]}
 (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
 trs="".join("<tr><td>%s</td><td>%d</td><td>%.2f</td><td>%.2f%%</td></tr>"%(html.escape(x["seed"]),x["trades"],x["profit_factor"] or 0,100*x["top3_gross_profit_share"] if x["top3_gross_profit_share"] is not None else 0) for x in rows)
 (out/"report.html").write_text("<!doctype html><html lang='ko'><meta charset='utf-8'><title>PRISM v4.86</title><style>body{background:#0b1729;color:#eaf4ff;font:15px system-ui;margin:30px}h1{color:#5de4bd}td,th{padding:10px;border-bottom:1px solid #345}</style><h1>PRISM v4.86 A/B Fragility</h1><p>Historical frozen diagnostic, not OOS.</p><table><tr><th>Seed</th><th>Trades</th><th>Profit factor</th><th>Top3 gross profit share</th></tr>"+trs+"</table></html>",encoding="utf-8")
 print(json.dumps({"status":"V486_COMPLETE","seeds":len(rows)}))
if __name__=="__main__":main()
