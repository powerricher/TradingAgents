"""PRISM FAST v4.2 — regime-aware expected-edge rolling walk-forward.

Pre-registered acceptance criteria (fixed before evaluating v4.2 OOS):
- pooled OOS profit factor >= 1.15
- pooled average D+1 return >= +0.10%
- >= 7 of 10 folds with PF > 1
- pooled -3% tail <= 6%

Architecture:
1) retain v4.1 downside model as a tail-risk gate;
2) add point-in-time market regime features;
3) fit expected D+1 return directly (regression);
4) estimate residual uncertainty from TRAINING data only;
5) rank by expected_return - uncertainty penalty and select only positive edge.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from .risk_gate import _prep, _X, stats
from .rolling_v41 import FOLDS

PASS_RULES={"min_profit_factor":1.15,"min_avg_return":0.001,"min_positive_folds":7,"max_minus3_rate":0.06}
REGIME_COLS=["regime_trend","regime_mom5","regime_vol","regime_bear","regime_riskoff"]
BASE_COLS=["compression","volume_anomaly","volume_accel","acceleration","breakout_proximity",
           "trend_quality","exhaustion_penalty","qqq_ret1","qqq_ret3","qqq_ret5","qqq_vs_ma20",
           "qqq_vol20","rel1","rel3","rel5","drawdown20","stock_vol20"]

def _auc(y,p):
 return float(roc_auc_score(y,p)) if len(np.unique(y))>1 else None

def _add_regime(df):
 z=df.copy()
 z["regime_trend"]=z["qqq_vs_ma20"].clip(-.15,.15)
 z["regime_mom5"]=z["qqq_ret5"].clip(-.15,.15)
 z["regime_vol"]=z["qqq_vol20"].clip(0,.10)
 z["regime_bear"]=((z["qqq_vs_ma20"]<0)&(z["qqq_ret5"]<0)).astype(float)
 z["regime_riskoff"]=((z["qqq_vs_ma20"]<-.03)|(z["qqq_ret5"]<-.06)|(z["qqq_vol20"]>.03)).astype(float)
 return z

def _XR(df):
 return df[BASE_COLS+REGIME_COLS].replace([np.inf,-np.inf],np.nan).fillna(0).astype(float)

def _regressor():
 return HistGradientBoostingRegressor(loss="huber",learning_rate=.045,max_iter=180,max_leaf_nodes=15,
                                      l2_regularization=2.0,min_samples_leaf=45,random_state=42)

def run(tickers,top_pct,out,uncertainty_penalty=.35):
 print("Preparing one cached 2021-2026 feature panel for v4.2...")
 panel=_add_regime(_prep(tickers,"2021-01-01","2026-10-06"))
 if panel.empty: raise RuntimeError("Feature panel is empty")
 panel["_date"]=pd.to_datetime(panel["date"])
 print(f"Panel ready: {len(panel):,} rows, {panel.ticker.nunique()} tickers")
 all_selected=[]; fold_meta=[]
 for j,(a,b,c,d) in enumerate(FOLDS,1):
  tr=panel[(panel._date>=pd.Timestamp(a))&(panel._date<=pd.Timestamp(b))].copy()
  te=panel[(panel._date>=pd.Timestamp(c))&(panel._date<=pd.Timestamp(d))].copy()
  if tr.empty or te.empty: raise RuntimeError(f"Fold {j} empty train/test")
  print(f"Fold {j}: train_n={len(tr):,}, test_n={len(te):,}")
  # Downside classifier is retained only as a tail-risk model.
  dn=LogisticRegression(max_iter=4000,class_weight="balanced").fit(_X(tr),tr.down)
  te["p_down"]=dn.predict_proba(_X(te))[:,1]
  # Expected-return model targets the quantity we actually monetize.
  reg=_regressor(); reg.fit(_XR(tr),tr.d1_return.astype(float))
  tr_pred=reg.predict(_XR(tr)); te["expected_return"]=reg.predict(_XR(te))
  # Point-in-time uncertainty: training residual scale by volatility quartile.
  resid=(tr.d1_return.to_numpy()-tr_pred)
  trv=pd.qcut(tr.stock_vol20.rank(method="first"),4,labels=False,duplicates="drop")
  scales={}
  for q in sorted(pd.Series(trv).dropna().unique()):
   rr=resid[np.asarray(trv)==q]; scales[int(q)]=float(np.sqrt(np.mean(rr**2))) if len(rr) else float(np.std(resid))
  cuts=tr.stock_vol20.quantile([.25,.5,.75]).to_numpy()
  qte=np.digitize(te.stock_vol20.to_numpy(),cuts,right=True)
  fallback=float(np.sqrt(np.mean(resid**2)))
  te["uncertainty"]=[scales.get(int(q),fallback) for q in qte]
  te["edge"]=te.expected_return-float(uncertainty_penalty)*te.uncertainty
  # Frozen tail-risk gate. Regime information belongs in alpha/edge, not ad-hoc threshold tuning.
  te["risk_pass"]=(te.p_down<=.45)&(te.qqq_vs_ma20>-.03)&(te.qqq_ret5>-.06)
  chosen=[]
  for dt,g in te[te.risk_pass].groupby("date"):
   n=max(1,int(np.ceil(len(g)*float(top_pct))))
   z=g.sort_values("edge",ascending=False).head(n).copy()
   z=z[z.edge>0]
   if not z.empty: chosen.append(z)
  sel=pd.concat(chosen,ignore_index=True) if chosen else te.iloc[0:0].copy()
  sel["fold"]=j; all_selected.append(sel)
  fold_meta.append({"fold":j,"train":[a,b],"test":[c,d],"train_n":len(tr),"test_n":len(te),
                    "down_auc":_auc(te.down,te.p_down),"all":stats(te),
                    "risk_pass":stats(te[te.risk_pass]),"selected":stats(sel)})
 pooled=pd.concat(all_selected,ignore_index=True) if all_selected else panel.iloc[0:0].copy()
 ps=stats(pooled); positive_folds=sum(1 for x in fold_meta if (x["selected"].get("profit_factor") or 0)>1)
 verdict={
  "profit_factor":bool((ps.get("profit_factor") or 0)>=PASS_RULES["min_profit_factor"]),
  "avg_return":bool(ps.get("avg_return",0)>=PASS_RULES["min_avg_return"]),
  "positive_folds":bool(positive_folds>=PASS_RULES["min_positive_folds"]),
  "minus3_tail":bool(ps.get("minus3_rate",1)<=PASS_RULES["max_minus3_rate"])
 }
 verdict["pass"]=all(verdict.values())
 meta={"model":"PRISM FAST v4.2 REGIME-AWARE EXPECTED EDGE","universe":tickers,
       "selection":"frozen v4 tail-risk gate; expected D+1 regression; uncertainty-adjusted positive edge; daily top percentile",
       "top_pct":float(top_pct),"uncertainty_penalty":float(uncertainty_penalty),
       "acceptance_rules":PASS_RULES,"positive_pf_folds":positive_folds,"verdict":verdict,
       "folds":fold_meta,"pooled_selected":ps}
 path=Path(out);path.parent.mkdir(parents=True,exist_ok=True)
 path.with_suffix(".json").write_text(json.dumps({"meta":meta,"selected_rows":pooled.to_dict("records")},indent=2,default=str),encoding="utf-8")
 path.with_name("rolling_summary.json").write_text(json.dumps(meta,indent=2,default=str),encoding="utf-8")
 rows="".join(f"<tr><td>{x['fold']}</td><td>{x['test'][0]}→{x['test'][1]}</td><td>{x['selected'].get('n',0)}</td><td>{x['selected'].get('avg_return',0):.2%}</td><td>{x['selected'].get('profit_factor') or 0:.2f}</td><td>{x['selected'].get('minus3_rate',0):.1%}</td><td>{x['down_auc'] or 0:.3f}</td></tr>" for x in fold_meta)
 status="PASS" if verdict["pass"] else "REJECT"
 html=f"""<!doctype html><meta charset='utf-8'><title>PRISM FAST v4.2</title><style>body{{font-family:Arial;background:#0b1020;color:#eef3ff;margin:32px}}.cards{{display:flex;gap:12px;flex-wrap:wrap}}.cards div{{background:#151d35;padding:16px;border-radius:12px}}table{{width:100%;border-collapse:collapse;margin-top:18px}}td,th{{padding:9px;border-bottom:1px solid #27314d;text-align:right}}td:nth-child(2),th:nth-child(2){{text-align:left}}</style><h1>PRISM FAST v4.2 — Regime-Aware Expected Edge</h1><p>Pre-registered verdict: <b>{status}</b> · PF≥1.15 · Avg D+1≥0.10% · ≥7/10 PF>1 · -3%≤6%</p><div class='cards'><div>Pooled N <b>{ps.get('n',0)}</b></div><div>Avg D+1 <b>{ps.get('avg_return',0):.2%}</b></div><div>PF <b>{ps.get('profit_factor') or 0:.2f}</b></div><div>-3% <b>{ps.get('minus3_rate',0):.1%}</b></div><div>PF>1 folds <b>{positive_folds}/10</b></div></div><table><tr><th>Fold</th><th>OOS period</th><th>N</th><th>Avg D+1</th><th>PF</th><th>-3%</th><th>Down AUC</th></tr>{rows}</table>"""
 path.write_text(html,encoding="utf-8"); return meta
