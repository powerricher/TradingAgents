"""PRISM FAST v2.2: true probability walk-forward calibration.

Train data fits P(D+1 >= +3%). Validation is scored with model.predict_proba;
no hand-scaled 0-100 score is used.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from .fast_lab import load_history, build_features, feature_parts, FEATURES

def frame(tickers,start,end):
    rows=[]; a,b=pd.Timestamp(start),pd.Timestamp(end)
    for ticker in tickers:
        df=build_features(load_history(ticker,start,end)).dropna()
        for i in range(len(df)-1):
            d=df.index[i].tz_localize(None) if getattr(df.index[i],"tzinfo",None) else df.index[i]
            if d<a or d>b: continue
            ret=float(df.iloc[i+1].Close/df.iloc[i].Close-1)
            rows.append({"ticker":ticker,"date":d.strftime("%Y-%m-%d"),
                         **feature_parts(df.iloc[i]),"target":int(ret>=.03),
                         "d1_return":ret,"entry_close":float(df.iloc[i].Close),
                         "exit_close":float(df.iloc[i+1].Close)})
    return pd.DataFrame(rows)

def _X(df):
    x=df[FEATURES].copy()
    x["exhaustion_penalty"]=100-x["exhaustion_penalty"]
    return x/100.0

def run(tickers,train_start,train_end,val_start,val_end,threshold,out):
    train=frame(tickers,train_start,train_end); val=frame(tickers,val_start,val_end)
    model=LogisticRegression(max_iter=3000)
    model.fit(_X(train),train.target)
    p=model.predict_proba(_X(val))[:,1]
    val=val.copy(); val["prob_plus3"]=p
    base=float(train.target.mean())
    val["edge_multiple"]=val.prob_plus3/base
    val["signal"]=pd.cut(val.prob_plus3,[-1,.15,.20,.25,2],
                         labels=["NORMAL","WATCH","STRONG","VERY_STRONG"])
    selected=val[val.prob_plus3>=float(threshold)].copy()
    def stats(z):
        if len(z)==0:return {"n":0}
        return {"n":int(len(z)),"plus3_hits":int(z.target.sum()),
                "plus3_precision":float(z.target.mean()),
                "positive_day_rate":float((z.d1_return>0).mean()),
                "avg_return":float(z.d1_return.mean()),
                "net_pnl_krw":int(round((z.d1_return*10_000_000).sum()))}
    dec=pd.qcut(val.prob_plus3,10,duplicates="drop")
    deciles=[]
    for interval,z in val.groupby(dec,observed=True):
        deciles.append({"prob_range":str(interval),"mean_pred":float(z.prob_plus3.mean()),
                        "actual_plus3":float(z.target.mean()),"n":int(len(z)),
                        "avg_return":float(z.d1_return.mean())})
    top={}
    for q in [.05,.10,.20]:
        cut=float(val.prob_plus3.quantile(1-q)); top[f"top_{int(q*100)}pct"]={"cutoff":cut,**stats(val[val.prob_plus3>=cut])}
    meta={"model":"PRISM LAB FAST v2.2 PROBABILITY","target":"D+1 >= +3%",
          "train":[train_start,train_end],"validation":[val_start,val_end],
          "train_rows":int(len(train)),"train_base_rate":base,
          "threshold_probability":float(threshold),
          "coefficients":dict(zip(FEATURES,[float(x) for x in model.coef_[0]])),
          "intercept":float(model.intercept_[0]),
          "validation":{"n":int(len(val)),"actual_plus3_rate":float(val.target.mean()),
                        "brier":float(brier_score_loss(val.target,p)),
                        "roc_auc":float(roc_auc_score(val.target,p)),
                        "all":stats(val),"selected":stats(selected),
                        "deciles":deciles,"top_slices":top}}
    path=Path(out);path.parent.mkdir(parents=True,exist_ok=True)
    path.with_suffix(".json").write_text(json.dumps({"meta":meta,"rows":val.to_dict("records")},indent=2),encoding="utf-8")
    path.with_name("calibration.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    rows="".join(f"<tr><td>{r.ticker}</td><td>{r.date}</td><td>{r.prob_plus3:.1%}</td><td>{r.edge_multiple:.2f}x</td><td>{r.signal}</td><td>{r.d1_return:.2%}</td><td>{'YES' if r.target else 'NO'}</td></tr>" for _,r in selected.iterrows())
    cards=f"<div class='cards'><div>Train base<br><b>{base:.1%}</b></div><div>Validation AUC<br><b>{meta['validation']['roc_auc']:.3f}</b></div><div>Brier<br><b>{meta['validation']['brier']:.4f}</b></div><div>Selected N<br><b>{len(selected)}</b></div><div>+3% precision<br><b>{stats(selected).get('plus3_precision',0):.1%}</b></div></div>"
    html=f"""<!doctype html><meta charset='utf-8'><title>PRISM FAST v2.2 Probability</title><style>body{{font-family:Arial;background:#0b1020;color:#e8eefc;margin:32px}}.cards{{display:flex;gap:12px;flex-wrap:wrap}}.cards div{{background:#151d35;padding:16px;border-radius:12px;min-width:140px}}table{{width:100%;border-collapse:collapse;margin-top:18px;background:#11182b}}th,td{{padding:9px;border-bottom:1px solid #27314d;text-align:right}}th:first-child,td:first-child{{text-align:left}}</style><h1>PRISM FAST v2.2 — Probability Walk-Forward</h1><p>P(D+1 ≥ +3%) · threshold ≥ {float(threshold):.0%} · no paid LLM/API</p>{cards}<table><tr><th>Ticker</th><th>Date</th><th>P(+3%)</th><th>Edge</th><th>Signal</th><th>D+1</th><th>+3%?</th></tr>{rows}</table>"""
    path.write_text(html,encoding="utf-8")
    return meta
