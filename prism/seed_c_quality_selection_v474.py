"""PRISM v4.74: expanded C1 quality ranking, fixed ex-ante scores.
No post-outcome training; historical challenger, NOT OOS.
"""
import json,html
from pathlib import Path
import numpy as np,pandas as pd
from .seed_c_multi_position_v471 import simulate
from .trend_v48 import load_prices
from .atr_frontier_v433 import CUTOFF
START=10_000_000.
def main():
 out=Path("results/prism-v474");out.mkdir(parents=True,exist_ok=True)
 df=pd.read_csv("results/prism-v473/candidate_exits.csv")
 prev=json.loads(Path("results/prism-v473/summary.json").read_text())
 equity=pd.read_csv("results/prism-v473/daily_equity.csv")
 base=df[df["mode"]=="BASE"].copy()
 expanded=df[df["mode"]=="COMPRESSION_80"].copy()
 required={"ticker","entry_date","exit_date","entry_price","exit_price","edge","reason","signal_rvol","prior_atr_pct","breakout_strength","gross_return"}
 if not required.issubset(expanded):raise RuntimeError("Missing v4.73 signal features")
 if expanded.duplicated(["ticker","entry_date"]).any() or base.duplicated(["ticker","entry_date"]).any():
  raise RuntimeError("Duplicate signal identity")
 keys=set(zip(base.ticker,base.entry_date))
 expanded["is_base"]=[(t,d) in keys for t,d in zip(expanded.ticker,expanded.entry_date)]
 if int(expanded.is_base.sum())!=len(base):raise RuntimeError("BASE not subset of expansion")
 # All components are known by signal close, not calibrated on future returns.
 # Fixed monotone score: capped RVOL, breakout percentage, prior ATR and base preference.
 expanded["quality_score"]=(
  30*np.clip((expanded.signal_rvol-1.8)/3,0,1)
  +30*np.clip(expanded.breakout_strength/.08,0,1)
  +20*np.clip((expanded.prior_atr_pct-.04)/.08,0,1)
  +20*expanded.is_base.astype(int))
 if expanded.quality_score.isna().any():raise RuntimeError("Invalid score")
 expanded["candidate_year"]=pd.to_datetime(expanded.entry_date).dt.year
 expanded["same_day_exit"]=expanded.entry_date==expanded.exit_date
 expanded["loss_candidate"]=expanded.gross_return<0
 expanded.to_csv(out/"candidate_quality_audit.csv",index=False)
 grouped=expanded.groupby(["candidate_year","is_base"]).agg(
  candidates=("ticker","size"),mean_gross=("gross_return","mean"),
  median_gross=("gross_return","median"),
  same_day=("same_day_exit","sum"),losses=("loss_candidate","sum")).reset_index()
 grouped.to_csv(out/"yearly_candidate_quality.csv",index=False)
 competition=expanded.groupby("entry_date").filter(lambda x:len(x)>1).copy()
 competition["rvol_rank"]=competition.groupby("entry_date")["signal_rvol"].rank(method="first",ascending=False)
 competition["quality_rank"]=competition.groupby("entry_date")["quality_score"].rank(method="first",ascending=False)
 competition.to_csv(out/"same_day_rank_competition.csv",index=False)
 ref=equity[equity["mode"]=="BASE"].sort_values("date")
 if ref.empty:raise RuntimeError("Missing v4.73 baseline curve")
 dates=[pd.Timestamp(x) for x in pd.to_datetime(ref.date).dt.normalize().unique()]
 prices={t:load_prices(t,"2021-01-01").loc[lambda x:x.index<=CUTOFF] for t in expanded.ticker.unique()}
 scenarios={"BASE_RVOL":base.assign(rank_score=base.edge),
  "EXPANDED_RVOL":expanded.assign(rank_score=expanded.signal_rvol),
  "EXPANDED_QUALITY":expanded.assign(rank_score=expanded.quality_score),
  "EXPANDED_BASE_PRIORITY":expanded.assign(rank_score=1000*expanded.is_base.astype(int)+expanded.quality_score)}
 report={"version":"v4.74","status":"HISTORICAL CHALLENGER NOT OOS","strategies":{},
  "new_candidates":int((~expanded.is_base).sum()),
  "limitations":[
   "No learned weights; quality score is an explicitly fixed research hypothesis.",
   "BASE candidates and 25%-by-four portfolio frozen; exit +8%/-4% H5.",
   "Quality uses completed signal bar only; BASE priority is not a proven predictive feature.",
   "Same-day stop-first OHLC fills, price data and current-listed historical universe remain biased.",
   "Multiple historical strategy comparisons make any apparent improvement in-sample.",
   "No orders, alerts, automatic production promotion or independent OOS."]}
 ledgers=[];curves=[]
 for name,z in scenarios.items():
  rows=z[["ticker","entry_date","exit_date","entry_price","exit_price","reason","rank_score"]].rename(columns={"rank_score":"edge"}).to_dict("records")
  perf,ledger,curve=simulate(rows,dates,prices,4,.25)
  report["strategies"][name]={**perf,"candidate_count":len(z),
   "mean_invested":float(np.mean([1-x["cash"]/x["equity"] for x in curve]))}
  for x in ledger:x["strategy"]=name
  for x in curve:x["strategy"]=name
  ledgers.extend(ledger);curves.extend(curve)
 for name,prior in (("BASE_RVOL","BASE"),("EXPANDED_RVOL","COMPRESSION_80")):
  v=report["strategies"][name]
  old=prev["strategies"][prior]
  if abs(v["final_krw"]-float(old["final_krw"]))>max(1,float(old["final_krw"])*1e-5) or v["trades"]!=old["trades"]:
   raise RuntimeError("Baseline replay mismatch: "+name)
 if len(report["strategies"])!=4:raise RuntimeError("Incomplete strategy frontier")
 pd.DataFrame(ledgers).to_csv(out/"ledger.csv",index=False)
 pd.DataFrame(curves).to_csv(out/"daily_equity.csv",index=False)
 (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
 lines="".join(f"<tr><td>{html.escape(k)}</td><td>{v['return']:.2%}</td><td>{v['mdd']:.2%}</td><td>{v['trades']}</td><td>{v['mean_invested']:.1%}</td></tr>" for k,v in report["strategies"].items())
 (out/"report.html").write_text("<!doctype html><html lang='ko'><meta charset='utf-8'><title>PRISM v4.74</title><style>body{background:#0c1729;color:#eaf4ff;font:15px system-ui;margin:30px}h1{color:#62e3b9}table{border-collapse:collapse;width:100%}td,th{padding:11px;border-bottom:1px solid #345}</style><h1>PRISM v4.74 · Compression Quality Selection</h1><p>Historical research only. Four policies, frozen exit and capital.</p><table><tr><th>Policy</th><th>Return</th><th>MDD</th><th>Trades</th><th>Invested</th></tr>"+lines+"</table></html>",encoding="utf-8")
 print(json.dumps({"status":"GENERATED","strategies":len(report["strategies"])}))
if __name__=="__main__":main()
