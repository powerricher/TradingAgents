"""PRISM v4.24: trade excursion / trend-efficiency ranking research.
Only completed earlier MA30 trades enter training; no future labels as features.
"""
import argparse,json,html
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from .ranking_v417 import FEATURES
from .trend_v48 import load_prices,portfolio

INPUTS=FEATURES+["contraction_ratio","range_contraction","volume_dryup"]
def main():
 p=argparse.ArgumentParser();p.add_argument("--scored",default="results/prism-v420/scored_signals.csv");p.add_argument("--out",default="results/prism-v424");a=p.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 df=pd.read_csv(a.scored);df.date=pd.to_datetime(df.date);df.entry_date=pd.to_datetime(df.entry_date);df.exit_date=pd.to_datetime(df.exit_date)
 df=df.sort_values(["date","ticker"]).reset_index(drop=True)
 if df[INPUTS+["ret","rank_VCP_ALL"]].isna().any().any():raise RuntimeError("Incomplete signal-time features")
 prices={t:load_prices(t,g.date.min()) for t,g in df.groupby("ticker")}
 labels=[]
 for r in df.itertuples(index=False):
  d=prices[r.ticker];e=d.index.get_loc(r.entry_date);x=d.index.get_loc(r.exit_date)
  if x<=e:raise RuntimeError("Nonpositive holding period")
  # Entry day high/low may occur before the opening fill, so only use highs/lows
  # starting from the following session; entry price is included as anchor.
  highs=np.r_[float(r.entry_price),d.High.iloc[e+1:x].to_numpy(dtype=float)]
  lows=np.r_[float(r.entry_price),d.Low.iloc[e+1:x].to_numpy(dtype=float)]
  mfe=float(np.max(highs)/r.entry_price-1)
  mae=float(np.min(lows)/r.entry_price-1)
  closes=d.Close.iloc[e:x].to_numpy(dtype=float)
  path=np.r_[float(r.entry_price),closes]
  efficiency=float((path[-1]-path[0])/max(np.abs(np.diff(path)).sum(),1e-8))
  labels.append((mfe,mae,efficiency))
 df[["mfe","mae","trend_efficiency"]]=np.asarray(labels)
 df["rank_EXCURSION"]=np.nan;df["train_exits"]=0
 for month,g in df.groupby(df.date.dt.to_period("M"),sort=True):
  tr=df[df.exit_date<month.start_time-pd.Timedelta(days=5)]
  if len(tr)<50:
   df.loc[g.index,"rank_EXCURSION"]=g.rank_VCP_ALL
   continue
  pred={}
  for target in ("mfe","mae","trend_efficiency","ret"):
   model=make_pipeline(StandardScaler(),Ridge(alpha=100.0))
   model.fit(tr[INPUTS],tr[target])
   pred[target]=model.predict(g[INPUTS])
  # Fixed provisional weights, not optimized on the historical outcomes.
  # MAE is negative; more negative MAE reduces score.
  score=.55*pred["ret"]+.25*pred["mfe"]+.15*pred["mae"]+.05*np.clip(pred["trend_efficiency"],-1,1)
  df.loc[g.index,"rank_EXCURSION"]=score
  df.loc[g.index,"train_exits"]=len(tr)
 if df.rank_EXCURSION.isna().any():raise RuntimeError("Missing scores")
 result={"version":"v4.24","status":"EXPLORATORY / NOT INDEPENDENT OOS","strategies":{},"ranking":{},"label_summary":{},
 "limitations":["MFE/MAE use future realized high/low only as training labels, never entry features.",
 "Entry-session high/low excluded to avoid intraday ordering ambiguity; exit-session extremes excluded because exit is at open.",
 "Efficiency is a close-to-close path diagnostic, not an executable return.",
 "Historical 80-stock universe and repeated experimentation are subject to selection bias.",
 "The score weights are provisional and must not be promoted without forward validation.",
 "Same fixed MA30 exits and one-position full-capital engine; no leverage.",
 "Daily-close MDD excludes intraday lows."]}
 for col in ("mfe","mae","trend_efficiency"):
  result["label_summary"][col]={"mean":float(df[col].mean()),"median":float(df[col].median())}
 ledgers=[];curves=[]
 for name,col in [("VCP_CONTROL","rank_VCP_ALL"),("EXCURSION_CHALLENGER","rank_EXCURSION")]:
  rows=[{"ticker":r.ticker,"mode":name,"entry_date":str(r.entry_date.date()),"exit_date":str(r.exit_date.date()),"entry_price":float(r.entry_price),"exit_price":float(r.exit_price),"edge":float(getattr(r,col)),"reason":r.reason} for r in df.itertuples(index=False)]
  perf,ledger,curve=portfolio(rows,prices,name,rotation=False)
  if perf["trades"]==0:raise RuntimeError("Invalid zero-trade simulation")
  result["strategies"][name]={"compound_return":perf["return"],"daily_close_mdd":perf["daily_close_mdd"],"trades":perf["trades"],"final_krw":perf["final_krw"]}
  result["ranking"][name]={"spearman":float(df[col].corr(df.ret,method="spearman")),"pearson":float(df[col].corr(df.ret))}
  for x in ledger:x["strategy"]=name
  for x in curve:x["strategy"]=name
  ledgers+=ledger;curves+=curve
 df.to_csv(out/"scored_signals.csv",index=False)
 pd.DataFrame(ledgers).to_csv(out/"ledger.csv",index=False)
 pd.DataFrame(curves).to_csv(out/"daily_equity.csv",index=False)
 (out/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
 rows="".join(f"<tr><td>{html.escape(k)}</td><td>{v['compound_return']:.2%}</td><td>{v['daily_close_mdd']:.2%}</td><td>{v['trades']}</td><td>{result['ranking'][k]['spearman']:.3f}</td></tr>" for k,v in result["strategies"].items())
 (out/"report.html").write_text("<!doctype html><meta charset='utf-8'><title>PRISM v4.24</title><style>body{background:#0b1729;color:#eaf4ff;font:15px system-ui;margin:28px}h1{color:#5de4bd}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #345}</style><h1>PRISM v4.24 — Excursion Selection Intelligence</h1><p>TECH80 · fixed MA30 · one position · historical research</p><table><tr><th>Model</th><th>Compound return</th><th>Daily MDD</th><th>Trades</th><th>Spearman</th></tr>"+rows+"</table><p>Research only, not independent OOS.</p>",encoding="utf-8")
 print(json.dumps({"status":"generated","strategies":result["strategies"]},ensure_ascii=False))
if __name__=="__main__":main()
