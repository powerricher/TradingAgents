"""PRISM v4.60: historical high proximity and trend audit.
Frozen A/B trades and all eligible signal candidates; no rule promotion.
"""
import json,html
from pathlib import Path
import numpy as np,pandas as pd
from .trend_v48 import load_prices
from .atr_frontier_v433 import CUTOFF
def feat(d,entry):
 dt=pd.Timestamp(entry)
 if dt not in d.index:return None
 i=d.index.get_loc(dt)-1
 if i<252:return None
 c=d.Close.astype(float);h=d.High.astype(float)
 px=float(c.iloc[i])
 z={"signal_close":px}
 for n in (20,50,252):
  high=float(h.iloc[i-n+1:i+1].max())
  z[f"high_{n}"]=high
  z[f"distance_{n}"]=px/high-1 if high>0 else None
 for n in (50,150,200):
  z[f"ma_{n}"]=float(c.iloc[i-n+1:i+1].mean())
 z["ma_aligned"]=bool(z["ma_50"]>z["ma_150"]>z["ma_200"])
 z["ma200_rising"]=bool(z["ma_200"]>float(c.iloc[i-219:i-19].mean()))
 return z
def main():
 out=Path("results/prism-v460");out.mkdir(parents=True,exist_ok=True)
 trades=pd.read_csv("results/prism-v459/trade_audit.csv")
 a=pd.read_csv("results/prism-v436/candidate_exits.csv")
 a=a[a.variant=="ATR4_CONTROL"].copy();a["seed"]="A"
 b=pd.read_csv("results/prism-v445/candidate_exits.csv")
 b=b[b.detector=="VCP_BREAKOUT"].copy();b["seed"]="B"
 required={"ticker","entry_date","entry_price","exit_price"}
 if any(not required.issubset(x) for x in (a,b,trades)):raise RuntimeError("Missing historical input columns")
 candidates=pd.concat([a,b],ignore_index=True)
 candidates=candidates.drop_duplicates(["seed","ticker","entry_date"])
 prices={};errors=[]
 for t in sorted(candidates.ticker.unique()):
  try:prices[t]=load_prices(t,"2020-01-01").loc[lambda x:x.index<=CUTOFF]
  except Exception as e:errors.append({"ticker":t,"error":str(e)})
 rows=[];missing=[]
 for r in candidates.itertuples(index=False):
  if r.ticker not in prices:missing.append({"seed":r.seed,"ticker":r.ticker,"entry_date":r.entry_date,"reason":"PRICE_UNAVAILABLE"});continue
  f=feat(prices[r.ticker],r.entry_date)
  if f is None:missing.append({"seed":r.seed,"ticker":r.ticker,"entry_date":r.entry_date,"reason":"NO_252_PRIOR_BARS"});continue
  rows.append({"seed":r.seed,"ticker":r.ticker,"entry_date":str(r.entry_date)[:10],
   "candidate_gross_return":float(r.exit_price/r.entry_price-1),**f})
 if not rows:raise RuntimeError("No high-proximity candidates")
 df=pd.DataFrame(rows)
 trades["entry_date"]=trades.entry_date.astype(str).str[:10]
 trades["seed"]=trades.seed.astype(str)
 selected=trades[["seed","ticker","entry_date","pnl","terminal_mark"]].drop_duplicates(["seed","ticker","entry_date"])
 df=df.merge(selected,on=["seed","ticker","entry_date"],how="left",validate="one_to_one")
 df["executed"]=df.pnl.notna()
 df["profitable_trade"]=df.pnl>0
 df["proximity_band"]=pd.cut(-df.distance_252,[-np.inf,.03,.07,.15,np.inf],
  labels=["0-3% or above high","3-7%","7-15%",">15%"])
 df.to_csv(out/"candidate_high_proximity.csv",index=False)
 df[df.executed].to_csv(out/"executed_trade_high_proximity.csv",index=False)
 pd.DataFrame(missing).to_csv(out/"missing_history.csv",index=False)
 (out/"price_errors.json").write_text(json.dumps(errors,ensure_ascii=False,indent=2))
 summary=[]
 for (seed,band),g in df.groupby(["seed","proximity_band"],observed=True):
  executed=g[g.executed]
  summary.append({"seed":seed,"band":str(band),"candidate_count":len(g),
   "executed_count":len(executed),"candidate_mean_gross_return":float(g.candidate_gross_return.mean()),
   "executed_win_rate":float(executed.profitable_trade.mean()) if len(executed) else None,
   "executed_mean_pnl":float(executed.pnl.mean()) if len(executed) else None,
   "ma_aligned_fraction":float(g.ma_aligned.mean())})
 pd.DataFrame(summary).to_csv(out/"proximity_calibration.csv",index=False)
 missed=[]
 for seed in ("A","B"):
  z=df[(df.seed==seed)&df.executed].sort_values("pnl",ascending=False)
  winners=z[z.pnl>0].head(5)
  for threshold in (.03,.07,.15):
   missed.append({"seed":seed,"max_distance":threshold,
    "top5_winners":len(winners),"excluded_top5":int((-winners.distance_252>threshold).sum()),
    "excluded_tickers":",".join(winners.loc[-winners.distance_252>threshold,"ticker"].astype(str))})
 pd.DataFrame(missed).to_csv(out/"winner_preservation.csv",index=False)
 report={"version":"v4.60","status":"HISTORICAL PROXIMITY AUDIT / NO NEW FILTER",
  "candidate_rows":len(df),"executed_rows":int(df.executed.sum()),"missing_history":len(missing),
  "price_errors":len(errors),"seed_a_executed":int(df[(df.seed=="A")&df.executed].shape[0]),
  "seed_b_executed":int(df[(df.seed=="B")&df.executed].shape[0]),
  "limitations":[
   "All high proximity and MA features use only completed session preceding entry.",
   "252 prior trading bars required; IPOs and missing history excluded, creating selection bias.",
   "Candidate gross returns are historical outcomes, not executable portfolio returns.",
   "Winning trade labels use legacy capital_after PnL, including possible terminal marks.",
   "High proximity is diagnostic only; no backtest of a proximity filter is claimed.",
   "Proximity band 0-3% includes prices above the prior 252-session high.",
   "Historical universe is retrospectively chosen and 2022-26 sample repeatedly inspected.",
   "No alerts, live trades, or changes to Seed A/B entry or exit rules."]}
 (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
 (out/"report.html").write_text("<!doctype html><meta charset='utf-8'><title>PRISM v4.60</title><style>body{background:#0b1729;color:#eaf4ff;font:15px system-ui;margin:28px}h1{color:#5de4bd}pre{white-space:pre-wrap}</style><h1>PRISM v4.60 · High Proximity Audit</h1><pre>"+html.escape(json.dumps(report,ensure_ascii=False,indent=2))+"</pre>",encoding="utf-8")
 print(json.dumps({"status":"generated","candidates":len(df),"executed":int(df.executed.sum())}))
if __name__=="__main__":main()
