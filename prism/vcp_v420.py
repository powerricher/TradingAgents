"""PRISM v4.20 VCP feature ablation and robustness.
Fixed TECH80 signals/MA30 exits; past-completed-trades training only.
"""
import argparse,json,html
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from .ranking_v417 import FEATURES
from .trend_v48 import portfolio,load_prices
VCP=["contraction_ratio","range_contraction","volume_dryup"]
VARIANTS={"RIDGE":FEATURES,"VCP_ALL":FEATURES+VCP,
 "VCP_VOLATILITY":FEATURES+[VCP[0]],
 "VCP_RANGE":FEATURES+[VCP[1]],
 "VCP_VOLUME":FEATURES+[VCP[2]]}
def main():
 p=argparse.ArgumentParser();p.add_argument("--scored",default="results/prism-v419/feature_scored_signals.csv");p.add_argument("--out",default="results/prism-v420");a=p.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 df=pd.read_csv(a.scored)
 df.date=pd.to_datetime(df.date);df.exit_date=pd.to_datetime(df.exit_date)
 df=df.sort_values(["date","ticker"]).reset_index(drop=True)
 cols=list(set(sum(VARIANTS.values(),[])))
 if df[cols+["ret"]].isna().any().any():raise ValueError("Incomplete point-in-time features")
 prices={}
 for ticker,g in df.groupby("ticker"):prices[ticker]=load_prices(ticker,g.date.min())
 report={"version":"v4.20","status":"HISTORICAL ABLATION / NOT INDEPENDENT OOS","variants":{},"ranking":{},"yearly":{},"concentration":{},
 "notes":["Same 2022-2026 history has been repeatedly researched; results are not independent OOS.",
 "All signals and MA30 exits are frozen; only contemporaneous rank order changes.",
 "Ridge is fit only on exits completed at least 5 calendar days before prediction month.",
 "Historical fixed 80-stock universe has survivorship bias.",
 "Daily-close MDD omits intraday lows."]}
 ledgers=[];curves=[]
 for name,features in VARIANTS.items():
  score=np.full(len(df),np.nan)
  for month,g in df.groupby(df.date.dt.to_period("M"),sort=True):
   train=df[df.exit_date<month.start_time-pd.Timedelta(days=5)]
   if len(train)<40:score[g.index]=g.edge.to_numpy();continue
   model=make_pipeline(StandardScaler(),Ridge(alpha=100.0))
   model.fit(train[features],train.ret)
   score[g.index]=model.predict(g[features])
  if not np.isfinite(score).all():raise RuntimeError("Invalid score "+name)
  df["rank_"+name]=score
  rows=[{"ticker":r.ticker,"mode":name,"entry_date":r.entry_date,"exit_date":str(r.exit_date.date()),
   "entry_price":float(r.entry_price),"exit_price":float(r.exit_price),"edge":float(score[i]),"reason":r.reason}
   for i,r in enumerate(df.itertuples(index=False))]
  perf,ledger,curve=portfolio(rows,prices,name,rotation=False)
  report["variants"][name]=perf
  report["ranking"][name]={"spearman":float(pd.Series(score).corr(df.ret,method="spearman")),
   "pearson":float(pd.Series(score).corr(df.ret))}
  lg=pd.DataFrame(ledger).sort_values("exit_date")
  lg["pnl"]=lg.capital_after.diff().fillna(lg.capital_after-10_000_000.)
  report["yearly"][name]={str(y):{"trades":len(g),"realized_pnl":float(g.pnl.sum())} for y,g in lg.groupby(lg.exit_date.str[:4])}
  wins=lg[lg.pnl>0].sort_values("pnl",ascending=False)
  report["concentration"][name]={"top3_winning_pnl_share":float(wins.head(3).pnl.sum()/wins.pnl.sum()) if len(wins) else None,
   "top5_winning_pnl_share":float(wins.head(5).pnl.sum()/wins.pnl.sum()) if len(wins) else None}
  for x in ledger:x["variant"]=name
  for x in curve:x["variant"]=name
  ledgers+=ledger;curves+=curve
 df.to_csv(out/"scored_signals.csv",index=False)
 pd.DataFrame(ledgers).to_csv(out/"ledger.csv",index=False)
 pd.DataFrame(curves).to_csv(out/"daily_equity.csv",index=False)
 (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
 trs="".join(f"<tr><td>{html.escape(k)}</td><td>{v['return']:.2%}</td><td>{v['daily_close_mdd']:.2%}</td><td>{report['ranking'][k]['spearman']:.3f}</td></tr>" for k,v in report["variants"].items())
 (out/"report.html").write_text("<!doctype html><meta charset='utf-8'><title>PRISM v4.20</title><style>body{background:#0b1729;color:#eaf4ff;font:15px system-ui;padding:28px}h1{color:#5eead4}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #345}</style><h1>PRISM v4.20 · VCP Feature Ablation</h1><p>Fixed TECH80 signals and MA30 exits; historical research only</p><table><tr><th>Feature Set</th><th>Compound Return</th><th>Daily MDD</th><th>Spearman</th></tr>"+trs+"</table><p>See JSON for yearly P&amp;L and concentration.</p>",encoding="utf-8")
 print(json.dumps({"status":"generated","variants":{k:v["return"] for k,v in report["variants"].items()}},ensure_ascii=False))
if __name__=="__main__":main()
