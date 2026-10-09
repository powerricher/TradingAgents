"""PRISM v4.26: chronological loss-rejection challenger.
Frozen TECH80/VCP candidates and MA30 exits; train on past completed trades only.
Reject signals, do not rotate. No future labels in features.
"""
import argparse,json,html
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression,Ridge
from .ranking_v417 import FEATURES
from .trend_v48 import load_prices,portfolio
FEATS=FEATURES+["contraction_ratio","range_contraction","volume_dryup"]
def run():
 p=argparse.ArgumentParser();p.add_argument("--scored",default="results/prism-v420/scored_signals.csv");p.add_argument("--out",default="results/prism-v426");a=p.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 df=pd.read_csv(a.scored)
 for k in ("date","entry_date","exit_date"):df[k]=pd.to_datetime(df[k])
 df=df.sort_values(["date","ticker"]).reset_index(drop=True)
 if df[FEATS+["ret","rank_VCP_ALL"]].isna().any().any():raise RuntimeError("Missing input fields")
 df["p_loss"]=np.nan;df["expected_ma30"]=np.nan;df["train_n"]=0
 for month,g in df.groupby(df.date.dt.to_period("M"),sort=True):
  train=df[df.exit_date<month.start_time-pd.Timedelta(days=5)]
  if len(train)<50 or (train.ret<=0).nunique()<2:
   df.loc[g.index,"p_loss"]=.5
   df.loc[g.index,"expected_ma30"]=g.rank_VCP_ALL
   continue
  x=train[FEATS];y=(train.ret<=0).astype(int)
  loss=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=3000))
  profit=make_pipeline(StandardScaler(),Ridge(alpha=100))
  loss.fit(x,y);profit.fit(x,train.ret)
  df.loc[g.index,"p_loss"]=loss.predict_proba(g[FEATS])[:,1]
  df.loc[g.index,"expected_ma30"]=profit.predict(g[FEATS])
  df.loc[g.index,"train_n"]=len(train)
 if df[["p_loss","expected_ma30"]].isna().any().any():raise RuntimeError("Unscored candidates")
 prices={t:load_prices(t,g.date.min()) for t,g in df.groupby("ticker")}
 report={"version":"v4.26","status":"HISTORICAL CHALLENGER / NOT INDEPENDENT OOS",
 "strategies":{},"gate_diagnostics":{},"limitations":[
 "Uses same historically researched 80-stock universe; results cannot establish future alpha.",
 "Frozen signals, fixed MA30 exits, no rotation, no leverage.",
 "Loss labels are realized MA30 trade outcomes, known only after exit; training excludes unclosed trades.",
 "Initial months use fallback probability 0.5; risk filter is disabled in fallback months.",
 "Fixed provisional loss thresholds 0.55 and 0.65 are research hypotheses, not tuned parameters.",
 "Gate selection removes candidates before rank competition; baseline retains all original eligible signals.",
 "All ranks are generated from signal-time features and prior completed trades.",
 "Daily-close MDD excludes intraday troughs."]}
 ledgers=[];curves=[]
 for name,threshold in [("VCP_HOLD",None),("LOSS_GATE_55",.55),("LOSS_GATE_65",.65)]:
  if threshold is None:chosen=df.copy()
  else:chosen=df[(df.train_n<50)|(df.p_loss<=threshold)].copy()
  rows=[{"ticker":r.ticker,"mode":name,"entry_date":str(r.entry_date.date()),"exit_date":str(r.exit_date.date()),
    "entry_price":float(r.entry_price),"exit_price":float(r.exit_price),"edge":float(r.rank_VCP_ALL),"reason":r.reason} for r in chosen.itertuples(index=False)]
  if not rows:raise RuntimeError("All signals rejected")
  perf,ledger,curve=portfolio(rows,prices,name,rotation=False)
  if perf["trades"]==0:raise RuntimeError("Zero executed trades")
  report["strategies"][name]={"return":perf["return"],"daily_close_mdd":perf["daily_close_mdd"],
    "trades":perf["trades"],"final_krw":perf["final_krw"],"candidate_signals":len(chosen),
    "rejected_signals":len(df)-len(chosen)}
  report["gate_diagnostics"][name]={"rejected_candidate_win_rate":float((df.loc[~df.index.isin(chosen.index),"ret"]>0).mean()) if len(chosen)<len(df) else None,
    "rejected_candidate_big_win_10pct":int((df.loc[~df.index.isin(chosen.index),"ret"]>=.10).sum()) if len(chosen)<len(df) else 0}
  for x in ledger:x["strategy"]=name
  for x in curve:x["strategy"]=name
  ledgers+=ledger;curves+=curve
 df.to_csv(out/"scored_signals.csv",index=False)
 pd.DataFrame(ledgers).to_csv(out/"ledger.csv",index=False)
 pd.DataFrame(curves).to_csv(out/"daily_equity.csv",index=False)
 (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
 rows="".join(f"<tr><td>{html.escape(k)}</td><td>{v['return']:.2%}</td><td>{v['daily_close_mdd']:.2%}</td><td>{v['trades']}</td><td>{v['rejected_signals']}</td></tr>" for k,v in report["strategies"].items())
 (out/"report.html").write_text("<!doctype html><meta charset='utf-8'><title>PRISM v4.26</title><style>body{background:#0b1729;color:#eaf4ff;font:15px system-ui;padding:28px}h1{color:#5de4bd}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #345}</style><h1>PRISM v4.26 · False Signal Rejection</h1><p>Frozen TECH80, MA30, one-position portfolio; historical research only</p><table><tr><th>Variant</th><th>Return</th><th>Daily MDD</th><th>Trades</th><th>Rejected signals</th></tr>"+rows+"</table>",encoding="utf-8")
 print(json.dumps({"status":"generated","strategies":report["strategies"]},ensure_ascii=False))
if __name__=="__main__":run()
