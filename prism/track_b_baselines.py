"""Track B v2: compare chronological probability and return models to train-only baselines."""
import argparse,json,html
from pathlib import Path
import numpy as np,pandas as pd
def main():
 p=argparse.ArgumentParser();p.add_argument("--predictions",default="results/prism-track-b/historical_predictions.csv");p.add_argument("--out",default="results/prism-track-b-v2");a=p.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 df=pd.read_csv(a.predictions);df["date"]=pd.to_datetime(df.date);df["exit_date"]=pd.to_datetime(df.exit_date)
 df=df.sort_values("date").reset_index(drop=True)
 # Training-only baseline: only exits already completed before prediction month.
 # Raw prediction file lacks early train labels; reconstruct using v4.17 scored history.
 full=pd.read_csv("results/prism-v417/scored_signals.csv")
 full["date"]=pd.to_datetime(full.date);full["exit_date"]=pd.to_datetime(full.exit_date)
 rows=[]
 for month,g in df.groupby(df.date.dt.to_period("M"),sort=True):
  start=month.start_time
  train=full[full.exit_date<start-pd.Timedelta(days=5)]
  if len(train)<50:continue
  prob=float((train.ret>0).mean());avg=float(train.ret.mean())
  z=g.copy();z["base_probability"]=prob;z["base_return"]=avg;rows.append(z)
 if not rows:raise RuntimeError("No baseline rows")
 x=pd.concat(rows,ignore_index=True);y=(x.ret>0).astype(float).to_numpy()
 metrics={}
 for name,prob,ret in [("MODEL",x.p_profit.to_numpy(),x.expected_return.to_numpy()),("TRAIN_ONLY_BASELINE",x.base_probability.to_numpy(),x.base_return.to_numpy())]:
  metrics[name]={"brier":float(np.mean((prob-y)**2)),"mae":float(np.mean(abs(ret-x.ret.to_numpy()))),"n":len(x)}
 metrics["delta_brier_model_minus_baseline"]=metrics["MODEL"]["brier"]-metrics["TRAIN_ONLY_BASELINE"]["brier"]
 metrics["delta_mae_model_minus_baseline"]=metrics["MODEL"]["mae"]-metrics["TRAIN_ONLY_BASELINE"]["mae"]
 result={"status":"HISTORICAL CHRONOLOGICAL RESEARCH / NOT INDEPENDENT HOLDOUT","metrics":metrics,
 "warnings":["Baseline uses historical trade exits known before each month; no future labels for baseline.",
 "The sample has been repeatedly researched; model selection here is not independent OOS.",
 "Probability calibration alone does not prove tradable portfolio alpha."]}
 x.to_csv(out/"predictions_with_baseline.csv",index=False)
 (out/"summary.json").write_text(json.dumps(result,indent=2))
 (out/"report.html").write_text("<!doctype html><meta charset='utf-8'><h1>PRISM Track B — Baseline Calibration</h1><pre>"+html.escape(json.dumps(result,indent=2))+"</pre>")
 print(json.dumps(metrics))
if __name__=="__main__":main()
