"""PRISM LAB FAST v2: anomaly/impulse-oriented deterministic backtest.

No paid LLM/API. Point-in-time price/volume features only.
v2 objective: detect pre-move asymmetry rather than reward already-extended trends.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from datetime import timedelta
import math
import pandas as pd
import yfinance as yf

@dataclass(frozen=True)
class FastSignal:
    ticker:str; date:str; score:float; decision:str; entry_close:float; exit_close:float
    d1_return:float; pnl_krw:int; features:dict
    def as_dict(self): return asdict(self)

def _clip(x,lo=0.,hi=100.): return max(lo,min(hi,float(x)))
def _rsi(c,n=14):
    d=c.diff(); up=d.clip(lower=0).rolling(n).mean(); dn=(-d.clip(upper=0)).rolling(n).mean()
    rs=up/dn.replace(0,math.nan); return 100-(100/(1+rs))

def load_history(ticker,start,end):
    s=pd.Timestamp(start)-timedelta(days=160); e=pd.Timestamp(end)+timedelta(days=14)
    x=yf.download(ticker,start=s.strftime("%Y-%m-%d"),end=e.strftime("%Y-%m-%d"),auto_adjust=False,progress=False)
    if x.empty: raise ValueError(f"No price history for {ticker}")
    if isinstance(x.columns,pd.MultiIndex): x.columns=x.columns.get_level_values(0)
    return x

def build_features(df):
    x=df.copy(); c=x.Close; v=x.Volume
    x["ret1"]=c.pct_change(); x["ret3"]=c.pct_change(3); x["ret5"]=c.pct_change(5); x["ret20"]=c.pct_change(20)
    x["ma20"]=c.rolling(20).mean(); x["ma60"]=c.rolling(60).mean()
    x["vol5"]=x.ret1.rolling(5).std(); x["vol20"]=x.ret1.rolling(20).std()
    x["compression"]=x.vol5/x.vol20
    x["rvol20"]=v/v.rolling(20).mean()
    x["vol_accel"]=x["rvol20"]/x["rvol20"].shift(3).rolling(3).mean()
    x["accel"]=x["ret3"]-(x["ret20"]/20*3)
    x["rsi14"]=_rsi(c)
    x["dist_ma20"]=c/x.ma20-1; x["dist_ma60"]=c/x.ma60-1
    x["range20_hi"]=c/c.rolling(20).max()
    return x

def score_row(r):
    # Reward pre-move compression + improving participation + controlled acceleration.
    compression=_clip(110-70*r["compression"])
    volume_anomaly=_clip(35+35*min(max(r["rvol20"],0),2.2))
    volume_accel=_clip(50+60*(r["vol_accel"]-1))
    acceleration=_clip(50+700*r["accel"])
    breakout_proximity=_clip(100-500*abs(1-r["range20_hi"]))
    trend_quality=_clip(50+350*r["dist_ma60"])
    # Explicitly penalize post-spike/overextended states.
    exhaustion=_clip(max(0,r["ret1"]-.035)*1200 + max(0,r["ret5"]-.09)*500 + max(0,r["dist_ma20"]-.10)*350)
    parts={"compression":compression,"volume_anomaly":volume_anomaly,"volume_accel":volume_accel,
           "acceleration":acceleration,"breakout_proximity":breakout_proximity,
           "trend_quality":trend_quality,"exhaustion_penalty":exhaustion}
    raw=(.18*compression+.16*volume_anomaly+.12*volume_accel+.18*acceleration+
         .14*breakout_proximity+.12*trend_quality+.10*(100-exhaustion))
    return round(_clip(raw),2),{k:round(float(v),3) for k,v in parts.items()}

def backtest_ticker(ticker,start,end,threshold=70.,capital_krw=10_000_000):
    df=build_features(load_history(ticker,start,end)).dropna(); a,b=pd.Timestamp(start),pd.Timestamp(end); out=[]
    for i in range(len(df)-1):
        date=df.index[i].tz_localize(None) if getattr(df.index[i],"tzinfo",None) else df.index[i]
        if date<a or date>b: continue
        score,features=score_row(df.iloc[i])
        if score<threshold: continue
        entry=float(df.iloc[i].Close); exit_=float(df.iloc[i+1].Close); ret=exit_/entry-1
        out.append(FastSignal(ticker,date.strftime("%Y-%m-%d"),score,"BUY",entry,exit_,ret,round(capital_krw*ret),features))
    return out
