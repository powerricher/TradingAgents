"""PRISM v4.13: frozen D+1 risk/return estimator with entry quality gate.
Compare 50-stock baseline against 80-stock baseline and 80-stock quality filter.
Research only. Signal-time features, no future outcomes used in selection.
"""
import argparse,json,html
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from .rolling_v42 import _add_regime,_XR,_regressor
from .risk_gate import _prep,_X
from .rolling_v41 import FOLDS
from .universe_v45 import UNIVERSE as CORE50
from .universe_v413 import UNIVERSE
def main():
 p=argparse.ArgumentParser();p.add_argument("--out",default="results/prism-v413");a=p.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 active=list(UNIVERSE)
 excluded=[]
 while True:
  try:
   panel=_add_regime(_prep(active,"2021-01-01","2026-10-06"))
   break
  except ValueError as exc:
   import re
   match=re.search(r"No price history for ([A-Z0-9.\\-]+)",str(exc))
   if not match or match.group(1) not in active:raise
   missing=match.group(1)
   active.remove(missing)
   excluded.append({"ticker":missing,"reason":"No historical price data"})
   print("DATA_COVERAGE_EXCLUSION",missing,flush=True)
   if len(active)<70:raise RuntimeError("Fewer than 70 tickers available")

 if panel.empty:raise RuntimeError("Empty price panel")
 panel["_date"]=pd.to_datetime(panel.date)
 output=[];diagnostics=[]
 for j,(a0,b,c,d) in enumerate(FOLDS,1):
  train=panel[(panel._date>=pd.Timestamp(a0))&(panel._date<=pd.Timestamp(b))].copy()
  test=panel[(panel._date>=pd.Timestamp(c))&(panel._date<=pd.Timestamp(d))].copy()
  if train.empty or test.empty:raise RuntimeError("Empty fold "+str(j))
  dn=LogisticRegression(max_iter=4000,class_weight="balanced").fit(_X(train),train.down)
  reg=_regressor().fit(_XR(train),train.d1_return)
  train_pred=reg.predict(_XR(train))
  residual=train.d1_return.to_numpy()-train_pred
  quartile=pd.qcut(train.stock_vol20.rank(method="first"),4,labels=False,duplicates="drop")
  scales={int(q):float(np.sqrt(np.mean(residual[np.asarray(quartile)==q]**2))) for q in sorted(pd.Series(quartile).dropna().unique())}
  cuts=train.stock_vol20.quantile([.25,.5,.75]).to_numpy()
  fallback=float(np.sqrt(np.mean(residual**2)))
  test["p_down"]=dn.predict_proba(_X(test))[:,1]
  test["expected_return"]=reg.predict(_XR(test))
  test["uncertainty"]=[scales.get(int(q),fallback) for q in np.digitize(test.stock_vol20.to_numpy(),cuts,right=True)]
  test["edge"]=test.expected_return-.35*test.uncertainty
  test["risk_pass"]=(test.p_down<=.45)&(test.qqq_vs_ma20>-.03)&(test.qqq_ret5>-.06)
  # Fixed, explicitly declared signal-time quality filter. No hindsight tuning.
  # rel5 positive, price drawdown within 10%, stock volatility below 5%.
  test["quality_pass"]=(test.rel5>0)&(test.drawdown20>-.10)&(test.stock_vol20<.05)
  # Market filter is applied after base selection, not refitted.
  test["market_pass"]=(test.qqq_vs_ma20>0)&(test.qqq_ret5>0)
  for variant,subset in [("CORE50",test[test.ticker.isin(CORE50)]),("TECH80",test),("TECH80_QUALITY",test)]:
   selected=[]
   for dt,g in subset[subset.risk_pass].groupby("date"):
    n=max(1,int(np.ceil(len(g)*.10)))
    top=g.nlargest(n,"edge")
    top=top[top.edge>0]
    if variant=="TECH80_QUALITY":
     top=top[top.quality_pass&top.market_pass]
    if not top.empty:selected.append(top)
   sel=pd.concat(selected) if selected else subset.iloc[:0].copy()
   sel=sel.copy();sel["fold"]=j;sel["variant"]=variant
   output.append(sel)
   diagnostics.append({"fold":j,"variant":variant,"selected":len(sel),"risk_pass":int(subset.risk_pass.sum()),"positive_edge_after_gate":int(((subset.edge>0)&subset.risk_pass).sum()),"market_pass":int(subset.market_pass.sum()),"quality_pass":int(subset.quality_pass.sum())})
 all_selected=pd.concat(output,ignore_index=True)
 cols=["ticker","date","fold","variant","edge","expected_return","uncertainty","p_down","qqq_vs_ma20","qqq_ret5","rel5","drawdown20","stock_vol20"]
 all_selected[cols].to_csv(out/"signals.csv",index=False)
 summary={"version":"v4.13","status":"RESEARCH / NOT INDEPENDENT OOS" if not excluded else "DEGRADED COVERAGE / NOT FULL 80","universe_size":len(active),"requested_universe_size":len(UNIVERSE),"universe":active,"excluded_tickers":excluded,"variants":{k:int(v) for k,v in all_selected.groupby("variant").size().items()},"fold_diagnostics":diagnostics,
 "selection_rules":{"TECH80_QUALITY":"top 10% by edge after risk gate, positive edge, rel5>0, drawdown20>-10%, stock_vol20<5%, QQQ above MA20 and QQQ 5D>0"},
 "limitations":["80-stock universe is retrospectively selected and includes companies with shorter trading history","Same historical sample has been repeatedly inspected; not independent validation","Fixed quality thresholds are exploratory, not calibrated on independent train periods","No portfolio or MA30 exit replay in this module","D+1 label uses close-to-next-close; execution at next open requires separate replay","No direct comparison of equal-model 50-stock training: CORE50 rows here use 80-stock fitted model"]}
 (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2))
 rows="".join(f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>" for k,v in summary["variants"].items())
 (out/"report.html").write_text("<!doctype html><meta charset='utf-8'><title>PRISM v4.13</title><style>body{background:#0c172a;color:#eaf4ff;font:15px system-ui;margin:28px}h1{color:#62e6c5}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #345}</style><h1>PRISM v4.13 · Tech 80 Entry Quality</h1><p>80 technology stocks, frozen D+1 architecture, MA30 exit to be replayed separately.</p><table><tr><th>Variant</th><th>Selected signals</th></tr>"+rows+"</table><p>These are signals, not portfolio returns. See summary.json for fold diagnostics and limitations.</p>",encoding="utf-8")
 print(json.dumps({"status":"generated","variants":summary["variants"]}))
if __name__=="__main__":main()
