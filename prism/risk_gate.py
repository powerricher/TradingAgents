"""PRISM FAST v4 — Risk Gate.
Adds market regime and relative-strength features, then uses downside probability
as a hard gate before ranking upside opportunity. No LLM/API required.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,brier_score_loss
from .fast_lab import load_history,build_features,feature_parts,FEATURES

V4_FEATURES=FEATURES+["qqq_ret1","qqq_ret3","qqq_ret5","qqq_vs_ma20","qqq_vol20",
                     "rel1","rel3","rel5","drawdown20","stock_vol20"]

def _prep(tickers,start,end):
    q=load_history("QQQ",start,end).copy()
    q["qqq_ret1"]=q.Close.pct_change();q["qqq_ret3"]=q.Close.pct_change(3);q["qqq_ret5"]=q.Close.pct_change(5)
    q["qqq_vs_ma20"]=q.Close/q.Close.rolling(20).mean()-1
    q["qqq_vol20"]=q.Close.pct_change().rolling(20).std()
    rows=[];a,b=pd.Timestamp(start),pd.Timestamp(end)
    for t in tickers:
        d=build_features(load_history(t,start,end)).copy()
        d["stock_ret1"]=d.Close.pct_change();d["stock_ret3"]=d.Close.pct_change(3);d["stock_ret5"]=d.Close.pct_change(5)
        d["stock_vol20"]=d.Close.pct_change().rolling(20).std()
        d["drawdown20"]=d.Close/d.Close.rolling(20).max()-1
        d=d.join(q[["qqq_ret1","qqq_ret3","qqq_ret5","qqq_vs_ma20","qqq_vol20"]],how="left").dropna()
        d["rel1"]=d.stock_ret1-d.qqq_ret1;d["rel3"]=d.stock_ret3-d.qqq_ret3;d["rel5"]=d.stock_ret5-d.qqq_ret5
        for i in range(len(d)-1):
            dt=d.index[i].tz_localize(None) if getattr(d.index[i],"tzinfo",None) else d.index[i]
            if dt<a or dt>b:continue
            ret=float(d.iloc[i+1].Close/d.iloc[i].Close-1);p=feature_parts(d.iloc[i])
            rows.append({"ticker":t,"date":dt.strftime("%Y-%m-%d"),**p,
              **{k:float(d.iloc[i][k]) for k in V4_FEATURES if k not in FEATURES},
              "d1_return":ret,"up":int(ret>=.03),"down":int(ret<=-.03)})
    return pd.DataFrame(rows)

def _X(z):
    x=z[V4_FEATURES].copy();x["exhaustion_penalty"]=100-x["exhaustion_penalty"]
    for k in FEATURES:x[k]=x[k]/100
    return x.replace([np.inf,-np.inf],np.nan).fillna(0)

def stats(z):
    if len(z)==0:return {"n":0}
    r=z.d1_return.to_numpy();pos=r[r>0].sum();neg=abs(r[r<0].sum())
    return {"n":int(len(z)),"avg_return":float(r.mean()),"positive_rate":float((r>0).mean()),
      "plus3_rate":float((r>=.03).mean()),"minus3_rate":float((r<=-.03).mean()),
      "profit_factor":float(pos/neg) if neg else None,"net_pnl_krw":int(round(r.sum()*1e7))}

def run(tickers,train_start,train_end,val_start,val_end,max_down,min_up,out):
    tr=_prep(tickers,train_start,train_end);va=_prep(tickers,val_start,val_end)
    up=LogisticRegression(max_iter=4000,class_weight="balanced").fit(_X(tr),tr.up)
    dn=LogisticRegression(max_iter=4000,class_weight="balanced").fit(_X(tr),tr.down)
    va=va.copy();va["p_up"]=up.predict_proba(_X(va))[:,1];va["p_down"]=dn.predict_proba(_X(va))[:,1]
    # Hard risk gate first. Opportunity ranking happens only after survival.
    va["risk_pass"]=(va.p_down<=float(max_down))&(va.qqq_vs_ma20>-0.03)&(va.qqq_ret5>-0.06)
    va["opportunity"]=va.p_up-va.p_down
    va["eligible"]=va.risk_pass&(va.p_up>=float(min_up))&(va.opportunity>0)
    sel=va[va.eligible].copy()
    top={}
    passed=va[va.risk_pass].sort_values("opportunity",ascending=False)
    for q in [.05,.10,.20]:
        n=max(1,int(np.ceil(len(passed)*q)));top[f"top_{int(q*100)}pct_after_gate"]=stats(passed.head(n))
    meta={"model":"PRISM FAST v4 RISK GATE","train":[train_start,train_end],"validation":[val_start,val_end],
      "features":V4_FEATURES,"train_rows":len(tr),"validation_rows":len(va),
      "thresholds":{"max_p_down":float(max_down),"min_p_up":float(min_up),"qqq_vs_ma20_floor":-0.03,"qqq_ret5_floor":-0.06},
      "metrics":{"up_auc":float(roc_auc_score(va.up,va.p_up)),"down_auc":float(roc_auc_score(va.down,va.p_down)),
       "up_brier":float(brier_score_loss(va.up,va.p_up)),"down_brier":float(brier_score_loss(va.down,va.p_down))},
      "all":stats(va),"risk_pass":stats(va[va.risk_pass]),"selected":stats(sel),"top_slices":top}
    path=Path(out);path.parent.mkdir(parents=True,exist_ok=True)
    path.with_suffix(".json").write_text(json.dumps({"meta":meta,"rows":va.to_dict("records")},indent=2),encoding="utf-8")
    path.with_name("risk_gate.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    rows="".join(f"<tr><td>{r.ticker}</td><td>{r.date}</td><td>{r.p_up:.1%}</td><td>{r.p_down:.1%}</td><td>{r.opportunity:.1%}</td><td>{r.qqq_ret5:.1%}</td><td>{r.d1_return:.2%}</td></tr>" for _,r in sel.sort_values(["date","opportunity"],ascending=[True,False]).iterrows())
    s=meta["selected"];html=f"""<!doctype html><meta charset='utf-8'><title>PRISM FAST v4</title><style>body{{font-family:Arial;background:#0b1020;color:#eef3ff;margin:32px}}.cards{{display:flex;gap:12px;flex-wrap:wrap}}.cards div{{background:#151d35;padding:16px;border-radius:12px}}table{{width:100%;border-collapse:collapse;margin-top:18px}}td,th{{padding:8px;border-bottom:1px solid #27314d;text-align:right}}td:first-child,th:first-child{{text-align:left}}</style><h1>PRISM FAST v4 — Risk Gate</h1><p>Downside hard gate → market regime gate → upside opportunity rank</p><div class='cards'><div>Selected <b>{s.get('n',0)}</b></div><div>Avg D+1 <b>{s.get('avg_return',0):.2%}</b></div><div>PF <b>{s.get('profit_factor') or 0:.2f}</b></div><div>-3% <b>{s.get('minus3_rate',0):.1%}</b></div></div><table><tr><th>Ticker</th><th>Date</th><th>P(+3)</th><th>P(-3)</th><th>Asymmetry</th><th>QQQ 5D</th><th>Actual</th></tr>{rows}</table>"""
    path.write_text(html,encoding="utf-8");return meta
