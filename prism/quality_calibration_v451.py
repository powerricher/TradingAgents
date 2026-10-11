"""PRISM v4.51 — Seed B signal quality calibration and opportunity audit.
Frozen v4.50 score, no parameter fitting, no changes to Seed A or exits.
"""
import argparse,json,html
from pathlib import Path
import numpy as np,pandas as pd
from .trend_v48 import load_prices
from .atr_frontier_v433 import CUTOFF
from .universe_v445 import UNIVERSE
HORIZONS=(3,5,10,20)
def future(d,entry):
 dt=pd.Timestamp(entry)
 if dt not in d.index:return None
 i=d.index.get_loc(dt)
 if i+max(HORIZONS)>=len(d):return None
 entry_open=float(d.Open.iloc[i])
 if not np.isfinite(entry_open) or entry_open<=0:return None
 out={}
 for n in HORIZONS:
  # D+n close return, and intraperiod adverse/favorable excursion
  window=d.iloc[i:i+n+1]
  out[f"ret_{n}"]=float(d.Close.iloc[i+n]/entry_open-1)
  out[f"mae_{n}"]=float(window.Low.min()/entry_open-1)
  out[f"mfe_{n}"]=float(window.High.max()/entry_open-1)
 return out
def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--scored",default="results/prism-v450/signal_time_quality.csv")
 ap.add_argument("--ledger",default="results/prism-v450/ledger.csv")
 ap.add_argument("--out",default="results/prism-v451")
 a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 c=pd.read_csv(a.scored);led=pd.read_csv(a.ledger)
 needed={"ticker","entry_date","detector","quality","edge"}
 if not needed.issubset(c):raise RuntimeError("Missing candidate score fields")
 prices={};errors=[]
 for ticker in sorted(set(c.ticker)):
  try:prices[ticker]=load_prices(ticker,"2021-01-01").loc[lambda x:x.index<=CUTOFF]
  except Exception as exc:errors.append({"ticker":ticker,"error":str(exc)})
 rows=[];unresolved=[]
 for r in c.itertuples(index=False):
  if r.ticker not in prices:unresolved.append({"ticker":r.ticker,"entry_date":r.entry_date,"reason":"missing_price"});continue
  x=future(prices[r.ticker],r.entry_date)
  if x is None:unresolved.append({"ticker":r.ticker,"entry_date":r.entry_date,"reason":"insufficient_horizon"});continue
  rows.append({"ticker":r.ticker,"entry_date":str(pd.Timestamp(r.entry_date).date()),
   "detector":r.detector,"quality":float(r.quality),"original_edge":float(r.edge),**x})
 if not rows:raise RuntimeError("No horizon-complete candidates")
 df=pd.DataFrame(rows)
 # Deduplicate repeated signal identity; the same candidate can be present
 # across different detectors, but not counted twice within a detector.
 df=df.drop_duplicates(["ticker","entry_date","detector"])
 df["year"]=pd.to_datetime(df.entry_date).dt.year
 df["period"]=np.where(df.year<=2024,"2022_2024_RESEARCH","2025_2026_LATER_SAMPLE")
 df["score_band"]=pd.cut(df.quality,[-np.inf,40,50,60,70,np.inf],
  labels=["<40","40-50","50-60","60-70",">=70"],right=False)
 df["failed_5d"]=(df.mae_5<=-.05)
 df["failed_10d"]=(df.mae_10<=-.08)
 df.to_csv(out/"candidate_forward_outcomes.csv",index=False)
 groups=[]
 for (det,period,band),g in df.groupby(["detector","period","score_band"],observed=True):
  item={"detector":det,"period":period,"score_band":str(band),"count":len(g),
   "failed_5d_share":float(g.failed_5d.mean()),"failed_10d_share":float(g.failed_10d.mean())}
  for n in (5,10,20):
   item[f"mean_ret_{n}"]=float(g[f"ret_{n}"].mean())
   item[f"median_ret_{n}"]=float(g[f"ret_{n}"].median())
   item[f"mean_mae_{n}"]=float(g[f"mae_{n}"].mean())
  groups.append(item)
 pd.DataFrame(groups).to_csv(out/"quality_calibration.csv",index=False)
 # Counterfactual rankings only where there are >=2 distinct tickers
 # available at the same entry open. This is not a portfolio backtest.
 rank_rows=[]
 for (det,day),g in df.groupby(["detector","entry_date"]):
  g=g.sort_values(["quality","ticker"],ascending=[False,True]).drop_duplicates("ticker")
  if len(g)<2:continue
  for rank,r in enumerate(g.head(3).itertuples(index=False),1):
   rank_rows.append({"detector":det,"entry_date":day,"rank":rank,"ticker":r.ticker,
    "quality":r.quality,"ret_5":r.ret_5,"ret_10":r.ret_10,"ret_20":r.ret_20,
    "mae_10":r.mae_10})
 ranks=pd.DataFrame(rank_rows)
 ranks.to_csv(out/"same_day_rank_alternatives.csv",index=False)
 # Identify historically executed trade identities without treating
 # hindsight winner removal as an investable filter.
 executed=led[led.experiment.astype(str).str.contains("TECH150_")].copy()
 executed=executed[executed.experiment.astype(str).str.endswith("_QUALITY_RANK")]
 keys=set(zip(executed.ticker,executed.entry_date.astype(str).str[:10]))
 df["executed_in_tech150_quality"]= [(r.ticker,r.entry_date) in keys for r in df.itertuples(index=False)]
 df[df.executed_in_tech150_quality].to_csv(out/"executed_forward_outcomes.csv",index=False)
 report={"version":"v4.51","status":"HISTORICAL DIAGNOSTIC / NOT INDEPENDENT OOS",
  "candidates_evaluated":len(df),"unresolved":len(unresolved),"download_errors":errors,
  "multi_candidate_rank_rows":len(ranks),"quality_gate55":"NOT_PROMOTED",
  "seed_a":"UNCHANGED","seed_b_trading_rules":"UNCHANGED",
  "limitations":[
   "D+3/5/10/20 future prices are used ONLY for retrospective evaluation, never for ranking.",
   "MAE/MFE are post-entry descriptive diagnostics; not executable stop fills.",
   "Score thresholds were developed using the same historical period; 2025-26 is later-period diagnostic, NOT untouched OOS.",
   "Same-day rank alternatives ignore opportunity blocking by open positions; not a portfolio return.",
   "Only historically available tickers with complete horizons included; missing coverage can bias calibration.",
   "2022-26 universe retrospectively selected, survivorship and multiple-testing biases remain.",
   "No new entry filter, fitted predictor, or change to actual trading rules has been adopted."]}
 pd.DataFrame(unresolved).to_csv(out/"unresolved.csv",index=False)
 (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
 (out/"report.html").write_text("<!doctype html><meta charset='utf-8'><title>PRISM v4.51</title><style>body{background:#0b1729;color:#eaf4ff;font:15px system-ui;margin:28px}h1{color:#5de4bd}pre{white-space:pre-wrap}</style><h1>PRISM v4.51 · Quality Calibration</h1><p>Retrospective diagnostic; no trading rule promoted.</p><pre>"+html.escape(json.dumps(report,ensure_ascii=False,indent=2))+"</pre>",encoding="utf-8")
 print(json.dumps(report,ensure_ascii=False))
if __name__=="__main__":main()
