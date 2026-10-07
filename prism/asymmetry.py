"""PRISM FAST v3 — Asymmetry Engine.
Three train-only models:
  1) P(D+1 >= +3%)
  2) P(D+1 <= -3%)
  3) E[D+1 return]
Validation is never used for fitting.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, roc_auc_score, mean_squared_error
from .calibrate import frame, _X
from .fast_lab import FEATURES

def _safe_auc(y,p):
    return float(roc_auc_score(y,p)) if len(np.unique(y))>1 else None

def _trade_stats(z):
    if len(z)==0:return {"n":0}
    r=z.d1_return.to_numpy()
    pos=r[r>0]; neg=r[r<0]
    return {"n":int(len(z)),
      "plus3_rate":float((r>=.03).mean()),"minus3_rate":float((r<=-.03).mean()),
      "positive_day_rate":float((r>0).mean()),"avg_return":float(r.mean()),
      "median_return":float(np.median(r)),
      "profit_factor":float(pos.sum()/abs(neg.sum())) if len(neg) and neg.sum()!=0 else None,
      "worst_trade":float(r.min()),"best_trade":float(r.max()),
      "net_pnl_krw":int(round(r.sum()*10_000_000))}

def run(tickers,train_start,train_end,val_start,val_end,min_edge,out):
    tr=frame(tickers,train_start,train_end); va=frame(tickers,val_start,val_end)
    Xtr,Xv=_X(tr),_X(va)
    yup=(tr.d1_return>=.03).astype(int); ydn=(tr.d1_return<=-.03).astype(int)
    up=LogisticRegression(max_iter=3000).fit(Xtr,yup)
    dn=LogisticRegression(max_iter=3000).fit(Xtr,ydn)
    ret=Ridge(alpha=10.0).fit(Xtr,tr.d1_return)
    va=va.copy()
    va["p_up3"]=up.predict_proba(Xv)[:,1]
    va["p_down3"]=dn.predict_proba(Xv)[:,1]
    va["expected_return"]=ret.predict(Xv)
    # Edge is intentionally interpretable: probability asymmetry plus expected-return term.
    va["asymmetry"]=va.p_up3-va.p_down3
    va["edge_score"]=va.asymmetry + 5.0*va.expected_return
    va["eligible"]=(va.p_up3>va.p_down3)&(va.expected_return>0)&(va.edge_score>=float(min_edge))
    chosen=va[va.eligible].copy()
    ranked=va.sort_values("edge_score",ascending=False)
    top={}
    for q in [.05,.10,.20]:
        n=max(1,int(np.ceil(len(ranked)*q))); z=ranked.head(n)
        top[f"top_{int(q*100)}pct"]=_trade_stats(z)
    # Sequential equity proxy: one independent 10M KRW trade per signal.
    pnl=chosen.sort_values(["date","ticker"]).d1_return.to_numpy()*10_000_000
    equity=np.cumsum(pnl); peak=np.maximum.accumulate(np.r_[0,equity])[1:]
    max_dd=float(np.min(equity-peak)) if len(equity) else 0.0
    meta={"model":"PRISM LAB FAST v3 ASYMMETRY","targets":["P(+3%)","P(-3%)","E[D+1]"],
      "train":[train_start,train_end],"validation":[val_start,val_end],
      "train_rows":int(len(tr)),"validation_rows":int(len(va)),
      "train_rates":{"plus3":float(yup.mean()),"minus3":float(ydn.mean()),"avg_return":float(tr.d1_return.mean())},
      "metrics":{"up_auc":_safe_auc((va.d1_return>=.03).astype(int),va.p_up3),
                 "down_auc":_safe_auc((va.d1_return<=-.03).astype(int),va.p_down3),
                 "up_brier":float(brier_score_loss((va.d1_return>=.03).astype(int),va.p_up3)),
                 "down_brier":float(brier_score_loss((va.d1_return<=-.03).astype(int),va.p_down3)),
                 "return_rmse":float(mean_squared_error(va.d1_return,va.expected_return)**.5)},
      "min_edge":float(min_edge),"all":_trade_stats(va),"selected":_trade_stats(chosen),
      "max_drawdown_krw_proxy":int(round(max_dd)),"top_slices":top,
      "coefficients":{"up":dict(zip(FEATURES,map(float,up.coef_[0]))),
                      "down":dict(zip(FEATURES,map(float,dn.coef_[0]))),
                      "return":dict(zip(FEATURES,map(float,ret.coef_)))}}
    path=Path(out);path.parent.mkdir(parents=True,exist_ok=True)
    path.with_suffix(".json").write_text(json.dumps({"meta":meta,"rows":va.to_dict("records")},indent=2),encoding="utf-8")
    path.with_name("asymmetry.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    show=chosen.sort_values(["date","edge_score"],ascending=[True,False])
    rows="".join(f"<tr><td>{r.ticker}</td><td>{r.date}</td><td>{r.p_up3:.1%}</td><td>{r.p_down3:.1%}</td><td>{r.expected_return:.2%}</td><td>{r.edge_score:.3f}</td><td>{r.d1_return:.2%}</td></tr>" for _,r in show.iterrows())
    s=meta["selected"]
    html=f"""<!doctype html><meta charset='utf-8'><title>PRISM FAST v3</title><style>body{{font-family:Arial;background:#0b1020;color:#e8eefc;margin:32px}}.cards{{display:flex;gap:12px;flex-wrap:wrap}}.cards div{{background:#151d35;padding:16px;border-radius:12px;min-width:145px}}table{{width:100%;border-collapse:collapse;margin-top:18px;background:#11182b}}th,td{{padding:9px;border-bottom:1px solid #27314d;text-align:right}}th:first-child,td:first-child{{text-align:left}}</style><h1>PRISM FAST v3 — Asymmetry Engine</h1><p>P(+3%) − P(-3%) + Expected Return · train-only fit / frozen validation</p><div class='cards'><div>Selected<br><b>{s.get('n',0)}</b></div><div>Avg D+1<br><b>{s.get('avg_return',0):.2%}</b></div><div>+3% hit<br><b>{s.get('plus3_rate',0):.1%}</b></div><div>-3% hit<br><b>{s.get('minus3_rate',0):.1%}</b></div><div>Profit factor<br><b>{s.get('profit_factor') or 0:.2f}</b></div><div>Max DD proxy<br><b>₩{meta['max_drawdown_krw_proxy']:,}</b></div></div><table><tr><th>Ticker</th><th>Date</th><th>P(+3%)</th><th>P(-3%)</th><th>E[D+1]</th><th>Edge</th><th>Actual D+1</th></tr>{rows}</table>"""
    path.write_text(html,encoding="utf-8")
    return meta
