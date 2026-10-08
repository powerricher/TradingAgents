"""PRISM v4.18: post-hoc robustness of frozen v4.17 trades and rankings.
No optimization, no independent holdout claim.
"""
import argparse,json,html
from pathlib import Path
import numpy as np,pandas as pd
def asfloat(x):
 return float(x) if pd.notna(x) else None
def analyze_ledger(g,scored,base_cost_bps=10):
 g=g.sort_values(["exit_date","entry_date"]).copy()
 g["prior_capital"]=g.capital_after.shift(1).fillna(10_000_000.)
 g["net_return"]=g.capital_after/g.prior_capital-1
 g["pnl"]=g.capital_after-g.prior_capital
 wins=g[g.pnl>0];loss=g[g.pnl<0]
 ordered=wins.sort_values("pnl",ascending=False)
 total_wins=float(wins.pnl.sum())
 concentration={str(n):{"win_pnl_share":float(ordered.head(n).pnl.sum()/total_wins) if total_wins else None,"top_pnl_krw":float(ordered.head(n).pnl.sum())} for n in (1,3,5)}
 years={}
 for year,q in g.groupby(g.exit_date.str[:4]):
  years[str(year)]={"trades":len(q),"realized_pnl_krw":float(q.pnl.sum()),"winning":int((q.pnl>0).sum()),"losing":int((q.pnl<0).sum())}
 # Cost stress applies extra round-trip bps per completed trade. The original
 # equity ledger already incorporates the original 10bps and stop assumptions.
 stress={}
 for bps in (10,20,30,50):
  extra=(bps-base_cost_bps)/10000.
  adjusted=np.clip(1+g.net_return.to_numpy()-extra,1e-12,None)
  equity=10_000_000.*np.cumprod(adjusted)
  peaks=np.maximum.accumulate(np.r_[10_000_000.,equity])[1:]
  stress[str(bps)]={"ending_capital_krw":float(equity[-1]),"cumulative_return":float(equity[-1]/10_000_000.-1),
    "closed_trade_mdd":float(np.min(equity/peaks-1))}
 return {"trades":len(g),"wins":len(wins),"losses":len(loss),"total_winning_pnl":total_wins,"total_losing_pnl":float(loss.pnl.sum()),
   "concentration":concentration,"yearly":years,"round_trip_cost_stress":stress}
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--ledger",default="results/prism-v417/ledger.csv")
 p.add_argument("--scored",default="results/prism-v417/scored_signals.csv")
 p.add_argument("--summary",default="results/prism-v417/summary.json")
 p.add_argument("--out",default="results/prism-v418")
 a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 ledger=pd.read_csv(a.ledger);scored=pd.read_csv(a.scored);prior=json.loads(Path(a.summary).read_text())
 required={"variant","ticker","entry_date","exit_date","capital_after"}
 if not required.issubset(ledger):raise ValueError("Missing ledger fields")
 results={"version":"v4.18","status":"POST-HOC ROBUSTNESS / NOT INDEPENDENT OOS","variants":{},"ranking":{},
 "cautions":["Cost sensitivity uses a completed-trade approximation; stop execution and trade ordering are not rerun.",
 "Closed-trade MDD under cost stress excludes unrealized equity drawdowns; original daily-close MDD remains in v4.17.",
 "Annual P&L is realized cash change, not time-weighted annual return.",
 "Ranking correlation uses all frozen candidate trades, not solely actual executed trades; labels are historical and overlapping.",
 "Retrospective universe and prior model-selection bias remain; forward validation is required."]}
 for name,g in ledger.groupby("variant"):
  results["variants"][name]=analyze_ledger(g,scored)
  results["variants"][name]["original_daily_close_mdd"]=prior["variants"][name]["daily_close_mdd"]
  results["variants"][name]["original_compound_return"]=prior["variants"][name]["return"]
 # Candidate-level correlation is a descriptive metric, not causal proof.
 if "ret" not in scored:raise ValueError("Missing realized MA30 returns")
 for name,col in [("EDGE_BASELINE","baseline_rank"),("PAST_EXIT_RIDGE","challenger_rank")]:
  x=scored[[col,"ret"]].dropna()
  results["ranking"][name]={"n":len(x),"spearman":asfloat(x[col].corr(x.ret,method="spearman")),
     "pearson":asfloat(x[col].corr(x.ret,method="pearson"))}
 # Test whether results depend on top winners: counterfactual zero out their
 # *realized* P&L, keeping transaction chronology fixed. Not a tradeable replay.
 for name,g in ledger.groupby("variant"):
  g=g.sort_values(["exit_date","entry_date"]).copy()
  ret=g.capital_after.to_numpy()/np.r_[10_000_000.,g.capital_after.to_numpy()[:-1]]-1
  for n in (1,3,5):
   idx=np.argsort(ret)[-n:]
   changed=ret.copy();changed[idx]=0.
   results["variants"][name][f"top_{n}_winners_zeroed_return"]=float(np.prod(1+changed)-1)
 (out/"summary.json").write_text(json.dumps(results,ensure_ascii=False,indent=2))
 rows="".join("<tr><td>"+html.escape(k)+"</td><td>"+f"{v['original_compound_return']:.2%}"+"</td><td>"+f"{v['original_daily_close_mdd']:.2%}"+"</td><td>"+str(v["trades"])+"</td><td>"+f"{results['ranking'][k]['spearman']:.3f}"+"</td></tr>" for k,v in results["variants"].items())
 (out/"report.html").write_text("<!doctype html><meta charset='utf-8'><title>PRISM v4.18</title><style>body{background:#0b1629;color:#e9f5ff;font:15px system-ui;margin:28px}h1{color:#5eead4}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #345}</style><h1>PRISM v4.18 · Ranking Robustness</h1><p>Post-hoc historical diagnostics. Not independent OOS.</p><table><tr><th>Ranker</th><th>Return</th><th>Daily-close MDD</th><th>Trades</th><th>Spearman</th></tr>"+rows+"</table><p>See summary.json for annual P&amp;L, top-winner dependence and cost stress.</p>",encoding="utf-8")
 print(json.dumps({"status":"generated","variants":{k:{"trades":v["trades"],"top_3_winners_zeroed_return":v["top_3_winners_zeroed_return"],"cost50bps":v["round_trip_cost_stress"]["50"]["cumulative_return"]} for k,v in results["variants"].items()}},ensure_ascii=False))
if __name__=="__main__":main()
