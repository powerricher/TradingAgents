"""PRISM v4.19: price-action feature ablation with frozen TECH80 signals.
These are research proxies, NOT verified rules from the four videos.
Event-day/AVWAP require timestamped event data and are explicitly deferred.
"""
import argparse,json,html
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from .ranking_v417 import FEATURES as BASE_FEATURES
from .trend_v48 import load_prices,portfolio

GROUPS={
 "BREAKOUT":["breakout_distance20","breakout_distance60","rs20","trend_persistence"],
 "VCP":["contraction_ratio","range_contraction","volume_dryup"],
 "AVWAP_PROXY":["rolling_vwap20_distance","rolling_vwap20_slope"],
}
def feature_frame(d):
 c=d.Close.astype(float);h=d.High.astype(float);l=d.Low.astype(float);v=d.Volume.astype(float)
 ret=c.pct_change()
 r20=(h-l).rolling(20,min_periods=20).mean()
 r10=(h-l).rolling(10,min_periods=10).mean()
 vw=(c*v).rolling(20,min_periods=20).sum()/v.rolling(20,min_periods=20).sum().replace(0,np.nan)
 f=pd.DataFrame(index=d.index)
 f["breakout_distance20"]=c/c.shift(1).rolling(20,min_periods=20).max()-1
 f["breakout_distance60"]=c/c.shift(1).rolling(60,min_periods=60).max()-1
 f["rs20"]=c.pct_change(20)
 f["trend_persistence"]=(ret>0).rolling(20,min_periods=20).mean()
 f["contraction_ratio"]=ret.rolling(5,min_periods=5).std()/ret.rolling(20,min_periods=20).std().replace(0,np.nan)
 f["range_contraction"]=r10/r20.replace(0,np.nan)
 f["volume_dryup"]=v.rolling(5,min_periods=5).mean()/v.rolling(20,min_periods=20).mean().replace(0,np.nan)
 f["rolling_vwap20_distance"]=c/vw-1
 f["rolling_vwap20_slope"]=vw/vw.shift(5)-1
 return f.replace([np.inf,-np.inf],np.nan)
def main():
 p=argparse.ArgumentParser();p.add_argument("--scored",default="results/prism-v417/scored_signals.csv");p.add_argument("--out",default="results/prism-v419");a=p.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 df=pd.read_csv(a.scored)
 df["date"]=pd.to_datetime(df.date).dt.normalize();df["exit_date"]=pd.to_datetime(df.exit_date)
 prices={};features=[];issues=[]
 for ticker,g in df.groupby("ticker"):
  try:
   d=load_prices(ticker,g.date.min());prices[ticker]=d
   f=feature_frame(d).reset_index()
   f=f.rename(columns={f.columns[0]:"date"})
   f["date"]=pd.to_datetime(f["date"]).dt.tz_localize(None).dt.normalize()
   f["ticker"]=ticker
   features.append(f)
  except Exception as exc:issues.append({"ticker":ticker,"error":str(exc)})
 if issues:raise RuntimeError("Incomplete price history "+json.dumps(issues))
 f=pd.concat(features,ignore_index=True)
 df=df.merge(f,on=["ticker","date"],how="left",validate="one_to_one")
 cols=BASE_FEATURES+sum(GROUPS.values(),[])
 if df[cols].isna().any().any():
  bad=df[df[cols].isna().any(axis=1)]
  raise RuntimeError("Missing point-in-time features: "+str(len(bad)))
 variants={"CONTROL_RIDGE":BASE_FEATURES,
 "BREAKOUT":BASE_FEATURES+GROUPS["BREAKOUT"],
 "VCP":BASE_FEATURES+GROUPS["VCP"],
 "VWAP_PROXY":BASE_FEATURES+GROUPS["AVWAP_PROXY"],
 "COMBINED":BASE_FEATURES+sum(GROUPS.values(),[])}
 report={"version":"v4.19","status":"RESEARCH ONLY / NOT INDEPENDENT OOS",
 "feature_groups":GROUPS,"variants":{},"notes":[
 "Breakout/VCP are original numeric proxies inspired by general concepts, not copied video rules.",
 "VWAP_PROXY is rolling 20-day close-volume weighted mean, NOT event-anchored intraday VWAP.",
 "SMB second-day and event AVWAP are DEFERRED until event timestamp and intraday data exist.",
 "All features are computed at signal close, with next-open entry.",
 "Only already-eligible frozen signals are reranked; gate and MA30 exit unchanged.",
 "Training uses only completed MA30 exits at least 5 calendar days before each prediction month.",
 "Same historical period used repeatedly for development, so results are not independent OOS.",
 "Historical fixed 80-stock universe has survivorship bias."]}
 df=df.sort_values(["date","ticker"]).reset_index(drop=True)
 ledger_rows=[];equity_rows=[]
 for name,columns in variants.items():
  scores=np.full(len(df),np.nan)
  for month,g in df.groupby(df.date.dt.to_period("M"),sort=True):
   train=df[df.exit_date<month.start_time-pd.Timedelta(days=5)]
   if len(train)<40:
    scores[g.index]=g.edge.to_numpy()
    continue
   model=make_pipeline(StandardScaler(),Ridge(alpha=100.0))
   model.fit(train[columns],train.ret)
   scores[g.index]=model.predict(g[columns])
  if not np.isfinite(scores).all():raise RuntimeError("Invalid ranks "+name)
  df["rank_"+name]=scores
  rows=[{"ticker":r.ticker,"mode":name,"entry_date":r.entry_date,"exit_date":str(r.exit_date.date()),
   "entry_price":float(r.entry_price),"exit_price":float(r.exit_price),"edge":float(scores[i]),"reason":r.reason}
   for i,r in enumerate(df.itertuples(index=False))]
  result,ledger,curve=portfolio(rows,prices,name,rotation=False)
  report["variants"][name]=result
  for x in ledger:x["variant"]=name
  for x in curve:x["variant"]=name
  ledger_rows+=ledger;equity_rows+=curve
 df.to_csv(out/"feature_scored_signals.csv",index=False)
 pd.DataFrame(ledger_rows).to_csv(out/"ledger.csv",index=False)
 pd.DataFrame(equity_rows).to_csv(out/"daily_equity.csv",index=False)
 (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
 trs="".join(f"<tr><td>{html.escape(k)}</td><td>{v['return']:.2%}</td><td>{v['daily_close_mdd']:.2%}</td><td>{v['trades']}</td></tr>" for k,v in report["variants"].items())
 (out/"report.html").write_text("<!doctype html><meta charset='utf-8'><title>PRISM v4.19</title><style>body{background:#0b1729;color:#edf7ff;font:15px system-ui;margin:28px}h1{color:#5de4bd}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #345}</style><h1>PRISM v4.19 · Price Action Feature Ablation</h1><p>TECH80 · fixed MA30 · historical research only</p><table><tr><th>Variant</th><th>Compound return</th><th>Daily MDD</th><th>Trades</th></tr>"+trs+"</table><p>Event day and anchored VWAP deferred; rolling VWAP proxy is not anchored VWAP.</p>",encoding="utf-8")
 print(json.dumps({"status":"generated","variants":report["variants"]},ensure_ascii=False))
if __name__=="__main__":main()
