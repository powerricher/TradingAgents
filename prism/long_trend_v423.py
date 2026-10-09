"""PRISM v4.23 long-trend persistence research.
Frozen TECH80 signals, MA30 exits; chronological completed-exit training.
1-slot VCP control vs persistence challenger. 2-slot control retained via v4.22.
"""
import argparse,json,html
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,LogisticRegression
from .ranking_v417 import FEATURES
from .trend_v48 import load_prices,portfolio
from .portfolio_v422 import dynamic

FEATURES_LONG=FEATURES+["contraction_ratio","range_contraction","volume_dryup"]
def main():
 p=argparse.ArgumentParser();p.add_argument("--scored",default="results/prism-v420/scored_signals.csv");p.add_argument("--out",default="results/prism-v423");a=p.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 df=pd.read_csv(a.scored);df.date=pd.to_datetime(df.date);df.entry_date=pd.to_datetime(df.entry_date);df.exit_date=pd.to_datetime(df.exit_date)
 df=df.sort_values(["date","ticker"]).reset_index(drop=True)
 if df[FEATURES_LONG+["ret","rank_VCP_ALL"]].isna().any().any():raise RuntimeError("Missing frozen features")
 prices={t:load_prices(t,g.date.min()) for t,g in df.groupby("ticker")}
 # All targets are historical outcomes and never enter contemporaneous predictors.
 durations=[];early_fail=[]
 for r in df.itertuples(index=False):
  d=prices[r.ticker]
  entry=pd.Timestamp(r.entry_date);exit_day=pd.Timestamp(r.exit_date)
  if entry not in d.index or exit_day not in d.index:raise RuntimeError("Incomplete OHLC")
  entry_i=d.index.get_loc(entry);exit_i=d.index.get_loc(exit_day)
  durations.append(exit_i-entry_i)
  # Label for diagnostic only: negative mark after five sessions (if available before exit).
  early_i=min(entry_i+5,exit_i-1) if exit_i>entry_i else entry_i
  early_fail.append(float(d.Close.iloc[early_i])/float(r.entry_price)-1<0)
 df["hold_sessions"]=durations;df["early_failure_5"]=early_fail
 for n in (10,20,40):df[f"persist_{n}"]=(df.hold_sessions>=n).astype(int)
 df["rank_LONG_TREND"]=np.nan;df["training_exits"]=0
 for month,g in df.groupby(df.date.dt.to_period("M"),sort=True):
  train=df[df.exit_date<month.start_time-pd.Timedelta(days=5)]
  if len(train)<50:
   df.loc[g.index,"rank_LONG_TREND"]=g.rank_VCP_ALL
   continue
  X=train[FEATURES_LONG];Y=train.ret
  reg=make_pipeline(StandardScaler(),Ridge(alpha=100.))
  reg.fit(X,Y)
  pred=reg.predict(g[FEATURES_LONG])
  # Penalize short-lived setups; fit only when both outcomes present.
  for horizon,weight in [(10,.015),(20,.02),(40,.015)]:
   y=train[f"persist_{horizon}"]
   if y.nunique()==2:
    clf=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=2500))
    clf.fit(X,y)
    pred+=weight*(clf.predict_proba(g[FEATURES_LONG])[:,1]-y.mean())
  if train.early_failure_5.nunique()==2:
   clf=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=2500))
   clf.fit(X,train.early_failure_5.astype(int))
   pred-=.02*(clf.predict_proba(g[FEATURES_LONG])[:,1]-train.early_failure_5.mean())
  df.loc[g.index,"rank_LONG_TREND"]=pred
  df.loc[g.index,"training_exits"]=len(train)
 if df.rank_LONG_TREND.isna().any():raise RuntimeError("Unscored candidates")
 result={"version":"v4.23","status":"EXPLORATORY / NOT INDEPENDENT OOS",
 "scope":"80-stock frozen signals; same MA30 exits; 10M KRW capital",
 "strategies":{},"calibration":{},"limitations":[
 "Persistence is labeled by actual holding duration of frozen MA30 trades; this is not independent forward validation.",
 "Early-failure label uses first five closes or the last pre-exit close if earlier; not an early-exit rule.",
 "Model coefficients and weights are exploratory; same historical period was used repeatedly in PRISM research.",
 "Historical 80-stock selection and price data have survivorship bias.",
 "Only eligible frozen signals reranked; missed upside outside Risk Gate remains unresolved.",
 "Daily close MDD excludes intraday drawdowns."]}
 ledger_rows=[];curve_rows=[]
 for label,col in [("VCP_1S","rank_VCP_ALL"),("LONG_TREND_1S","rank_LONG_TREND")]:
  rows=[{"ticker":r.ticker,"mode":label,"entry_date":str(r.entry_date.date()),"exit_date":str(r.exit_date.date()),"entry_price":float(r.entry_price),"exit_price":float(r.exit_price),"edge":float(getattr(r,col)),"reason":r.reason} for r in df.itertuples(index=False)]
  perf,ledger,curve=portfolio(rows,prices,label,rotation=False)
  if perf["trades"]==0 or len(ledger)==0:
   raise RuntimeError("Invalid zero-trade portfolio: check entry_date normalization")
  result["strategies"][label]={"compound_return":perf["return"],"daily_close_mdd":perf["daily_close_mdd"],"trades":perf["trades"],"final_krw":perf["final_krw"]}
  for x in ledger:x["strategy"]=label
  for x in curve:x["strategy"]=label
  ledger_rows+=ledger;curve_rows+=curve
  result["calibration"][label]={"spearman":float(df[col].corr(df.ret,method="spearman"))}
 # Risk-control benchmark is fixed VCP_ALL dynamic 2 slots, weighted.
 records=df[["ticker","entry_date","exit_date","entry_price","exit_price","reason","rank_VCP_ALL"]].rename(columns={"rank_VCP_ALL":"rank"})
 perf,ledger,curve=dynamic(records,prices,2,True)
 if perf["trades"]==0:raise RuntimeError("Invalid zero-trade dynamic benchmark")
 result["strategies"]["VCP_DYNAMIC_2S_WEIGHTED"]={"compound_return":perf["compound_return"],"daily_close_mdd":perf["daily_close_mdd"],"trades":perf["trades"],"final_krw":perf["final_krw"]}
 for x in ledger:x["strategy"]="VCP_DYNAMIC_2S_WEIGHTED"
 for x in curve:x["strategy"]="VCP_DYNAMIC_2S_WEIGHTED"
 ledger_rows+=ledger;curve_rows+=curve
 df.to_csv(out/"scored_signals.csv",index=False)
 pd.DataFrame(ledger_rows).to_csv(out/"ledger.csv",index=False)
 pd.DataFrame(curve_rows).to_csv(out/"daily_equity.csv",index=False)
 (out/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
 trs="".join(f"<tr><td>{html.escape(k)}</td><td>{v['compound_return']:.2%}</td><td>{v['daily_close_mdd']:.2%}</td><td>{v['trades']}</td></tr>" for k,v in result["strategies"].items())
 (out/"report.html").write_text("<!doctype html><meta charset='utf-8'><title>PRISM v4.23</title><style>body{background:#0b1729;color:#eaf4ff;font:15px system-ui;margin:28px}h1{color:#5de4bd}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #345}</style><h1>PRISM v4.23 — Long Trend Persistence</h1><p>Historical research; not independent OOS. Frozen MA30 exit.</p><table><tr><th>Model</th><th>Compound return</th><th>Daily MDD</th><th>Trades</th></tr>"+trs+"</table>",encoding="utf-8")
 print(json.dumps({"status":"generated","strategies":result["strategies"]},ensure_ascii=False))
if __name__=="__main__":main()
