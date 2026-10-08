"""Track B: chronological MA30 trade-outcome prediction diagnostics.
Only settled historical exits may train; no future labels in features.
"""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression,Ridge
from .ranking_v417 import FEATURES
def main():
 p=argparse.ArgumentParser();p.add_argument("--scored",default="results/prism-v417/scored_signals.csv");p.add_argument("--out",default="results/prism-track-b");a=p.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 df=pd.read_csv(a.scored)
 df.date=pd.to_datetime(df.date);df.exit_date=pd.to_datetime(df.exit_date)
 if df[FEATURES+["ret"]].isna().any().any():raise ValueError("Incomplete training features")
 predictions=[];folds=[]
 for month,g in df.groupby(df.date.dt.to_period("M"),sort=True):
  start=month.start_time
  train=df[df.exit_date<start-pd.Timedelta(days=5)]
  if len(train)<50 or len(np.unique(train.ret>0))<2:continue
  x=train[FEATURES].to_numpy();y=train.ret.to_numpy()
  clf=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=3000))
  reg=make_pipeline(StandardScaler(),Ridge(alpha=100))
  clf.fit(x,y>0);reg.fit(x,y)
  pred=g[["ticker","date","ret","exit_date"]].copy()
  pred["p_profit"]=clf.predict_proba(g[FEATURES])[:,1]
  pred["expected_return"]=reg.predict(g[FEATURES])
  pred["month"]=str(month)
  predictions.append(pred)
  folds.append({"month":str(month),"training_exits":len(train),"predictions":len(g)})
 if not predictions:raise RuntimeError("No chronological evaluation months")
 allpred=pd.concat(predictions,ignore_index=True)
 allpred.to_csv(out/"historical_predictions.csv",index=False)
 # Descriptive metrics only; no claim of independent OOS after iterative research.
 y=(allpred.ret>0).astype(int)
 brier=float(np.mean((allpred.p_profit-y)**2))
 mae=float(np.mean(np.abs(allpred.expected_return-allpred.ret)))
 report={"track":"B","status":"HISTORICAL CHALLENGER / NOT INDEPENDENT OOS","n":len(allpred),"brier":brier,"return_mae":mae,"months":folds,
 "warnings":["Uses already researched 2022-2026 sample","No new trade entry ranking or portfolio replay yet","All training rows exited before prediction month minus 5 calendar days","Selection restricted to prior TECH80 positive-Edge signals"]}
 (out/"summary.json").write_text(json.dumps(report,indent=2))
 (out/"report.html").write_text("<!doctype html><meta charset='utf-8'><h1>PRISM Track B — MA30 Profit Probability</h1><p>Historical chronological research only</p><pre>"+json.dumps(report,indent=2)+"</pre>")
 print(json.dumps({"status":"generated","n":len(allpred),"brier":brier,"mae":mae}))
if __name__=="__main__":main()
