"""PRISM v4.17 chronological MA30-aligned ranking challenger.
Train only on COMPLETED earlier MA30 trades, predict returns for future signals.
No use of target trade outcome to select that same trade.
"""
import argparse,json,html
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from .trend_v48 import portfolio,load_prices

FEATURES=["edge","expected_return","uncertainty","p_down","qqq_vs_ma20","qqq_ret5","rel5","drawdown20","stock_vol20"]
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--signals",default="results/prism-v413/signals.csv")
 p.add_argument("--trades",default="results/prism-v414/trades.csv")
 p.add_argument("--out",default="results/prism-v417")
 a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 signals=pd.read_csv(a.signals)
 signals=signals[signals.variant=="TECH80"].copy()
 trades=pd.read_csv(a.trades)
 trades=trades[trades.variant=="TECH80"].copy()
 if len(signals)!=len(trades):raise RuntimeError("Incomplete frozen signal/trade coverage")
 signals["date"]=pd.to_datetime(signals.date)
 trades["signal_date"]=pd.to_datetime(trades.signal_date)
 trades["exit_date"]=pd.to_datetime(trades.exit_date)
 merged=signals.merge(trades[["ticker","signal_date","entry_date","exit_date","entry_price","exit_price","ret","reason"]],left_on=["ticker","date"],right_on=["ticker","signal_date"],validate="one_to_one")
 if len(merged)!=len(signals):raise RuntimeError("Signal matching incomplete")
 if merged[FEATURES].isna().any().any():raise RuntimeError("Missing signal-time features")
 merged=merged.sort_values(["date","ticker"]).reset_index(drop=True)
 merged["baseline_rank"]=merged.edge
 merged["challenger_rank"]=np.nan
 merged["train_count"]=0
 merged["ranking_status"]="BASELINE_FALLBACK"
 # Chronological retraining every calendar month; use only fully exited trades
 # at least 5 calendar days before each scoring month.
 for month,g in merged.groupby(merged.date.dt.to_period("M"),sort=True):
  start=month.start_time
  train=merged[merged.exit_date < start-pd.Timedelta(days=5)]
  ix=g.index
  if len(train)<40:
   merged.loc[ix,"challenger_rank"]=g.edge
   continue
  x=train[FEATURES].to_numpy(dtype=float)
  y=train.ret.to_numpy(dtype=float)
  # Fixed regularization; no validation tuning on these historical results.
  model=make_pipeline(StandardScaler(),Ridge(alpha=100.0))
  model.fit(x,y)
  pred=model.predict(g[FEATURES].to_numpy(dtype=float))
  # Keep original gate/positive edge decisions unchanged; only rerank
  # simultaneous frozen eligible signals.
  merged.loc[ix,"challenger_rank"]=pred
  merged.loc[ix,"train_count"]=len(train)
  merged.loc[ix,"ranking_status"]="PAST_EXITS_RIDGE"
 if merged.challenger_rank.isna().any():raise RuntimeError("Unranked signals")
 price={};errors=[]
 for ticker,g in merged.groupby("ticker"):
  try:price[ticker]=load_prices(ticker,g.date.min())
  except Exception as e:errors.append({"ticker":ticker,"error":str(e)})
 if errors:raise RuntimeError("Price coverage incomplete "+json.dumps(errors))
 summary={"version":"v4.17","status":"EXPLORATORY / NO INDEPENDENT HOLDOUT",
 "scope":"TECH80 fixed 346 signals, MA30 exits, 10bps round-trip, 20bps stop-touch stress",
 "train_rule":"Only previously completed trades at least five calendar days before each prediction month; Ridge alpha 100; baseline fallback under 40 examples",
 "variants":{},"limitations":["The same historical period has informed prior strategy development; not independent OOS.",
 "Training samples are selected by original positive Edge and Risk Gate, so challenger cannot discover previously excluded winners.",
 "MA30 realized trade returns are target labels; chronological exit-date separation prevents direct target leakage.",
 "Only same-day priority changes; no new entry dates or new signals.",
 "Retrospectively selected 80-stock universe may have survivorship bias.",
 "Daily close MDD omits intraday troughs; cost and stop assumptions are synthetic."]}
 all_ledgers=[];all_curves=[]
 for name,rank in [("EDGE_BASELINE","baseline_rank"),("PAST_EXIT_RIDGE","challenger_rank")]:
  rows=[]
  for r in merged.itertuples(index=False):
   rows.append({"ticker":r.ticker,"mode":name,"entry_date":r.entry_date,"exit_date":str(r.exit_date.date()),
   "entry_price":float(r.entry_price),"exit_price":float(r.exit_price),"edge":float(getattr(r,rank)),"reason":r.reason})
  result,ledger,curve=portfolio(rows,price,name,rotation=False)
  summary["variants"][name]=result
  for x in ledger:x["variant"]=name
  for x in curve:x["variant"]=name
  all_ledgers+=ledger;all_curves+=curve
 merged.to_csv(out/"scored_signals.csv",index=False)
 pd.DataFrame(all_ledgers).to_csv(out/"ledger.csv",index=False)
 pd.DataFrame(all_curves).to_csv(out/"daily_equity.csv",index=False)
 (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2))
 rows="".join("<tr><td>"+html.escape(k)+"</td><td>"+f"{v['return']:.2%}"+"</td><td>"+f"{v['daily_close_mdd']:.2%}"+"</td><td>"+str(v["trades"])+"</td></tr>" for k,v in summary["variants"].items())
 (out/"report.html").write_text("<!doctype html><meta charset='utf-8'><title>PRISM v4.17</title><style>body{background:#0b1629;color:#eaf5ff;font:15px system-ui;padding:28px}h1{color:#5eead4}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #345}</style><h1>PRISM v4.17 · MA30-Aligned Ranking</h1><p>Same TECH80 signals · chronological training on past closed trades · fixed exits</p><table><tr><th>Ranker</th><th>Compound return</th><th>Daily close MDD</th><th>Trades</th></tr>"+rows+"</table><p>Research only, not independent OOS. See summary.json for limitations.</p>",encoding="utf-8")
 print(json.dumps({"status":"generated","variants":summary["variants"]},ensure_ascii=False))
if __name__=="__main__":main()
