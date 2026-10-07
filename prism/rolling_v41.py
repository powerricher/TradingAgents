"""PRISM FAST v4.1 — Expanding rolling walk-forward.
Architecture is frozen from v4. Each fold fits only on data preceding its test
window. Selection is relative: risk gate, then top opportunity percentile.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from .risk_gate import _prep,_X,stats

FOLDS=[
 ("2021-01-01","2021-12-31","2022-01-01","2022-06-30"),
 ("2021-01-01","2022-06-30","2022-07-01","2022-12-31"),
 ("2021-01-01","2022-12-31","2023-01-01","2023-06-30"),
 ("2021-01-01","2023-06-30","2023-07-01","2023-12-31"),
 ("2021-01-01","2023-12-31","2024-01-01","2024-06-30"),
 ("2021-01-01","2024-06-30","2024-07-01","2024-12-31"),
 ("2021-01-01","2024-12-31","2025-01-01","2025-06-30"),
 ("2021-01-01","2025-06-30","2025-07-01","2025-12-31"),
 ("2021-01-01","2025-12-31","2026-01-01","2026-06-30"),
 ("2021-01-01","2026-06-30","2026-07-01","2026-10-06"),
]
def _auc(y,p):
 return float(roc_auc_score(y,p)) if len(np.unique(y))>1 else None
def run(tickers,top_pct,out):
 all_selected=[]; fold_meta=[]
 # Download/build the full point-in-time feature panel ONCE. Repeated Yahoo
 # downloads per fold caused rate-limit/empty-history failures.
 print("Preparing one cached 2021-2026 feature panel...")
 panel=_prep(tickers,"2021-01-01","2026-10-06")
 if panel.empty:
  raise RuntimeError("Feature panel is empty")
 panel["_date"]=pd.to_datetime(panel["date"])
 print(f"Panel ready: {len(panel):,} rows, {panel.ticker.nunique()} tickers")
 for j,(a,b,c,d) in enumerate(FOLDS,1):
  print(f"Fold {j}: train {a}..{b} | test {c}..{d}")
  tr=panel[(panel._date>=pd.Timestamp(a))&(panel._date<=pd.Timestamp(b))].copy()
  te=panel[(panel._date>=pd.Timestamp(c))&(panel._date<=pd.Timestamp(d))].copy()
  if tr.empty or te.empty:
   raise RuntimeError(f"Fold {j} has empty train/test: train={len(tr)}, test={len(te)}")
  print(f"Fold {j}: train_n={len(tr):,}, test_n={len(te):,}")
  up=LogisticRegression(max_iter=4000,class_weight="balanced").fit(_X(tr),tr.up)
  dn=LogisticRegression(max_iter=4000,class_weight="balanced").fit(_X(tr),tr.down)
  te=te.copy();te["p_up"]=up.predict_proba(_X(te))[:,1];te["p_down"]=dn.predict_proba(_X(te))[:,1]
  te["opportunity"]=te.p_up-te.p_down
  # Frozen v4 market/downside gate; ranking threshold is relative within each date.
  te["risk_pass"]=(te.p_down<=.45)&(te.qqq_vs_ma20>-.03)&(te.qqq_ret5>-.06)
  chosen=[]
  for dt,g in te[te.risk_pass].groupby("date"):
   n=max(1,int(np.ceil(len(g)*float(top_pct))))
   z=g.sort_values("opportunity",ascending=False).head(n).copy()
   z=z[z.opportunity>0]
   chosen.append(z)
  sel=pd.concat(chosen,ignore_index=True) if chosen else te.iloc[0:0].copy()
  sel["fold"]=j;all_selected.append(sel)
  fold_meta.append({"fold":j,"train":[a,b],"test":[c,d],"train_n":len(tr),"test_n":len(te),
    "up_auc":_auc(te.up,te.p_up),"down_auc":_auc(te.down,te.p_down),
    "all":stats(te),"risk_pass":stats(te[te.risk_pass]),"selected":stats(sel)})
 pooled=pd.concat(all_selected,ignore_index=True) if all_selected else pd.DataFrame()
 meta={"model":"PRISM FAST v4.1 ROLLING WALK-FORWARD","universe":tickers,
       "selection":"v4 risk gate then daily top opportunity percentile; opportunity > 0",
       "top_pct":float(top_pct),"folds":fold_meta,"pooled_selected":stats(pooled)}
 path=Path(out);path.parent.mkdir(parents=True,exist_ok=True)
 path.with_suffix(".json").write_text(json.dumps({"meta":meta,"selected_rows":pooled.to_dict("records")},indent=2,default=str),encoding="utf-8")
 path.with_name("rolling_summary.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
 rows="".join(f"<tr><td>{x['fold']}</td><td>{x['test'][0]}→{x['test'][1]}</td><td>{x['selected'].get('n',0)}</td><td>{x['selected'].get('avg_return',0):.2%}</td><td>{x['selected'].get('profit_factor') or 0:.2f}</td><td>{x['selected'].get('minus3_rate',0):.1%}</td><td>{x['down_auc'] or 0:.3f}</td></tr>" for x in fold_meta)
 s=meta["pooled_selected"]
 html=f"""<!doctype html><meta charset='utf-8'><title>PRISM FAST v4.1 Rolling</title><style>body{{font-family:Arial;background:#0b1020;color:#eef3ff;margin:32px}}.cards{{display:flex;gap:12px;flex-wrap:wrap}}.cards div{{background:#151d35;padding:16px;border-radius:12px}}table{{width:100%;border-collapse:collapse;margin-top:18px}}td,th{{padding:9px;border-bottom:1px solid #27314d;text-align:right}}td:nth-child(2),th:nth-child(2){{text-align:left}}</style><h1>PRISM FAST v4.1 — Rolling Walk-Forward</h1><p>Frozen Risk Gate · daily top {float(top_pct):.0%} opportunity · expanding train window</p><div class='cards'><div>Pooled N <b>{s.get('n',0)}</b></div><div>Avg D+1 <b>{s.get('avg_return',0):.2%}</b></div><div>PF <b>{s.get('profit_factor') or 0:.2f}</b></div><div>-3% <b>{s.get('minus3_rate',0):.1%}</b></div></div><table><tr><th>Fold</th><th>OOS period</th><th>N</th><th>Avg D+1</th><th>PF</th><th>-3%</th><th>Down AUC</th></tr>{rows}</table>"""
 path.write_text(html,encoding="utf-8");return meta
