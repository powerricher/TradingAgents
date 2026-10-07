"""Walk-forward calibration for PRISM FAST v2.1.

Fits feature weights ONLY on train-period observations to target P(D+1 >= +3%).
Validation dates never enter fitting. Uses sklearn logistic regression.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from .fast_lab import load_history,build_features,feature_parts,FEATURES,backtest_ticker
from .fast_report import write_report

def training_frame(tickers,start,end):
    rows=[]
    a,b=pd.Timestamp(start),pd.Timestamp(end)
    for ticker in tickers:
        df=build_features(load_history(ticker,start,end)).dropna()
        for i in range(len(df)-1):
            d=df.index[i].tz_localize(None) if getattr(df.index[i],"tzinfo",None) else df.index[i]
            if d<a or d>b: continue
            parts=feature_parts(df.iloc[i])
            ret=float(df.iloc[i+1].Close/df.iloc[i].Close-1)
            rows.append({"ticker":ticker,"date":d.strftime("%Y-%m-%d"),**parts,"target":int(ret>=.03),"d1_return":ret})
    return pd.DataFrame(rows)

def fit_weights(train):
    X=train[FEATURES].copy()
    # Exhaustion is intrinsically adverse; transform to quality so all coefficients remain interpretable.
    X["exhaustion_penalty"]=100-X["exhaustion_penalty"]
    model=LogisticRegression(max_iter=2000,class_weight="balanced")
    model.fit(X/100.0,train.target)
    coef=dict(zip(FEATURES,model.coef_[0]))
    # Convert logit contribution to a stable 0-100 ranking score around 50.
    scale=7.5
    weights={k:float(v*scale/100.0) for k,v in coef.items()}
    # because score_row receives raw penalty, invert fitted quality coefficient
    weights["exhaustion_penalty"]=-weights["exhaustion_penalty"]
    weights["_intercept"]=50.0
    return weights,model

def run(tickers,train_start,train_end,val_start,val_end,threshold,out):
    train=training_frame(tickers,train_start,train_end)
    weights,model=fit_weights(train)
    signals=[]
    for t in tickers: signals.extend(backtest_ticker(t,val_start,val_end,threshold,weights=weights))
    signals.sort(key=lambda x:(x.date,x.ticker))
    summary=write_report(signals,out)
    meta={"model":"PRISM LAB FAST v2.1 WALK-FORWARD","target":"D+1 >= +3%",
          "train":[train_start,train_end],"validation":[val_start,val_end],
          "train_rows":len(train),"train_positive_rate":float(train.target.mean()),
          "weights":weights,"validation_summary":summary}
    p=Path(out).with_name("calibration.json");p.write_text(json.dumps(meta,indent=2),encoding="utf-8")
    return meta
