"""PRISM v4.29: freeze 80-trained model vs 100-trained model, same selection.
Retrospective universe: historical model selection and survivorship bias.
"""
import argparse,json,html,re
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from .rolling_v42 import _add_regime,_XR,_regressor
from .risk_gate import _prep,_X
from .rolling_v41 import FOLDS
from .universe_v429 import CORE80,UNIVERSE
from .replay_v412 import replay
from .trend_v48 import portfolio

def fit_scores(panel,training_universe):
 selected=[];diagnostics=[];all_candidates=[]
 for fold,(a,b,c,d) in enumerate(FOLDS,1):
  train=panel[(panel.date>=a)&(panel.date<=b)&panel.ticker.isin(training_universe)].copy()
  test=panel[(panel.date>=c)&(panel.date<=d)].copy()
  if train.empty or test.empty:raise RuntimeError("Empty fold "+str(fold))
  dn=LogisticRegression(max_iter=4000,class_weight="balanced").fit(_X(train),train.down)
  reg=_regressor().fit(_XR(train),train.d1_return)
  residual=train.d1_return.to_numpy()-reg.predict(_XR(train))
  quartile=pd.qcut(train.stock_vol20.rank(method="first"),4,labels=False,duplicates="drop")
  scales={int(q):float(np.sqrt(np.mean(residual[np.asarray(quartile)==q]**2))) for q in sorted(pd.Series(quartile).dropna().unique())}
  cuts=train.stock_vol20.quantile([.25,.5,.75]).to_numpy()
  fallback=float(np.sqrt(np.mean(residual**2)))
  test["p_down"]=dn.predict_proba(_X(test))[:,1]
  test["expected_return"]=reg.predict(_XR(test))
  test["uncertainty"]=[scales.get(int(q),fallback) for q in np.digitize(test.stock_vol20.to_numpy(),cuts,right=True)]
  test["edge"]=test.expected_return-.35*test.uncertainty
  test["risk_pass"]=(test.p_down<=.45)&(test.qqq_vs_ma20>-.03)&(test.qqq_ret5>-.06)
  test["fold"]=fold
  all_candidates.append(test[["ticker","date","fold","edge","risk_pass","p_down","expected_return"]].copy())
  for universe,label in [(CORE80,"CORE80"),(UNIVERSE,"TECH100")]:
   subset=test[test.ticker.isin(universe)&test.risk_pass]
   picks=[]
   for day,g in subset.groupby("date"):
    n=max(1,int(np.ceil(len(g)*.10)))
    x=g.nlargest(n,"edge")
    x=x[x.edge>0]
    if not x.empty:picks.append(x)
   s=pd.concat(picks,ignore_index=True) if picks else subset.iloc[:0].copy()
   s["universe"]=label;selected.append(s)
   diagnostics.append({"fold":fold,"universe":label,"signals":len(s),"risk_pass":len(subset)})
 return pd.concat(selected,ignore_index=True),pd.concat(all_candidates,ignore_index=True),diagnostics

def main():
 p=argparse.ArgumentParser();p.add_argument("--out",default="results/prism-v429");a=p.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 active=list(UNIVERSE);excluded=[]
 while True:
  try:panel=_add_regime(_prep(active,"2021-01-01","2026-10-06"));break
  except ValueError as exc:
   m=re.search(r"No price history for ([A-Z0-9.\\-]+)",str(exc))
   if not m or m.group(1) not in active:raise
   active.remove(m.group(1));excluded.append(m.group(1))
   if len(active)<95:raise RuntimeError("Insufficient price coverage; refuse to label 100-stock comparison")
 if excluded:raise RuntimeError("Not full 100-stock universe: "+str(excluded))
 if panel.empty:raise RuntimeError("Empty panel")
 summary={"version":"v4.29","status":"HISTORICAL RESEARCH / NOT INDEPENDENT OOS","universe_count":len(active),
 "new_tickers":UNIVERSE[80:],"experiments":{},"ceiling":{},"diagnostics":{},
 "limitations":["Both universes are retrospectively selected and subject to survivorship bias.",
 "80-trained model is refit on historical 80 only, then applied to both universes.",
 "100-trained model uses the expanded training set; same fold boundaries.",
 "Same-day ex-post oracle is a hindsight heuristic, not an attainable upper bound.",
 "This stage compares D+1 entry signals and fixed MA30 exits; VCP reranking on TECH100 is not implemented.",
 "Historical sample has been repeatedly inspected; not independent validation.",
 "Candidate MA30 trades must be fully resolved or the workflow fails."]}
 all_trades=[];all_ledgers=[];all_curves=[]
 for training,label in [(CORE80,"FROZEN80_TRAIN"),(UNIVERSE,"REFIT100_TRAIN")]:
  signals,candidates,diag=fit_scores(panel,training)
  summary["diagnostics"][label]=diag
  candidates["training"]=label
  candidates.to_csv(out/(label+"_candidate_scores.csv.gz"),index=False,compression="gzip")
  for universe in ("CORE80","TECH100"):
   x=signals[signals.universe==universe].copy()
   if x.empty:raise RuntimeError("No signals "+label+universe)
   x[["ticker","date","edge","fold"]].to_csv(out/(label+"_"+universe+"_signals.csv"),index=False)
   key=label+"_"+universe
   perf,ledger,curve,trades=replay(x,out,key)
   if perf.get("status")!="COMPLETE" or perf.get("trades",0)==0:raise RuntimeError("Invalid replay "+key+" "+str(perf))
   summary["experiments"][key]=perf
   for r in ledger:r["experiment"]=key
   for r in curve:r["experiment"]=key
   for r in trades:r["experiment"]=key
   all_trades+=trades;all_ledgers+=ledger;all_curves+=curve
   # An ex-post ranking diagnostic over EXACT SAME frozen signals. Each
   # candidate's future MA30 trade return is used only in hindsight ranking.
   hindsight=[dict(r,edge=float(r["ret"])) for r in trades]
   from .trend_v48 import load_prices
   prices={t:load_prices(t,g.date.min()) for t,g in x.groupby("ticker")}
   oracle,_,_=portfolio(hindsight,prices,"HINDSIGHT",rotation=False)
   summary["ceiling"][key]={"hindsight_rank_compound_return":oracle["return"],
     "observed_return":perf["return"],
     "interpretation":"Greedy hindsight rank among already eligible signals, NOT theoretical market maximum or feasible prediction"}
 pd.DataFrame(all_trades).to_csv(out/"trades.csv",index=False)
 pd.DataFrame(all_ledgers).to_csv(out/"ledger.csv",index=False)
 pd.DataFrame(all_curves).to_csv(out/"daily_equity.csv",index=False)
 (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2))
 rows="".join(f"<tr><td>{html.escape(k)}</td><td>{v['return']:.2%}</td><td>{v['daily_close_mdd']:.2%}</td><td>{v['trades']}</td><td>{summary['ceiling'][k]['hindsight_rank_compound_return']:.2%}</td></tr>" for k,v in summary["experiments"].items())
 (out/"report.html").write_text("<!doctype html><meta charset='utf-8'><title>PRISM v4.29</title><style>body{background:#0b1729;color:#eaf4ff;font:15px system-ui;margin:28px}h1{color:#5de4bd}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #345}</style><h1>PRISM v4.29 · Universe 100 &amp; Opportunity Ceiling</h1><p>D+1 entry baseline / MA30; NOT yet a VCP 100-stock replay. Historical research only.</p><table><tr><th>Experiment</th><th>Return</th><th>MDD</th><th>Trades</th><th>Hindsight rank heuristic</th></tr>"+rows+"</table>",encoding="utf-8")
 print(json.dumps({"status":"generated","experiments":{k:v["return"] for k,v in summary["experiments"].items()}},ensure_ascii=False))
if __name__=="__main__":main()
