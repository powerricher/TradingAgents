"""PRISM LAB FAST: zero-paid-LLM deterministic historical backtest.

Uses public Yahoo price history through yfinance. No OpenAI/LLM API is called.
M1 purpose: rapidly test whether a stable, point-in-time score has monotonic
relationship with next-session returns before adding expensive qualitative agents.
"""

from __future__ import annotations
from dataclasses import asdict, dataclass
from datetime import timedelta
import math

import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class FastSignal:
    ticker: str
    date: str
    score: float
    decision: str
    entry_close: float
    exit_close: float
    d1_return: float
    pnl_krw: int
    features: dict

    def as_dict(self): return asdict(self)


def _clip(x, lo=0.0, hi=100.0):
    return max(lo, min(hi, float(x)))


def _rsi(close: pd.Series, n=14):
    d=close.diff(); up=d.clip(lower=0).rolling(n).mean(); dn=(-d.clip(upper=0)).rolling(n).mean()
    rs=up/dn.replace(0, math.nan)
    return 100-(100/(1+rs))


def load_history(ticker: str, start: str, end: str) -> pd.DataFrame:
    s=pd.Timestamp(start)-timedelta(days=120)
    e=pd.Timestamp(end)+timedelta(days=14)
    df=yf.download(ticker,start=s.strftime("%Y-%m-%d"),end=e.strftime("%Y-%m-%d"),
                   auto_adjust=False,progress=False)
    if df.empty: raise ValueError(f"No price history for {ticker}")
    if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    x=df.copy()
    c=x["Close"]; v=x["Volume"]
    x["ret1"]=c.pct_change()
    x["ret5"]=c.pct_change(5)
    x["ret20"]=c.pct_change(20)
    x["ma20"]=c.rolling(20).mean()
    x["ma60"]=c.rolling(60).mean()
    x["vol20"]=x["ret1"].rolling(20).std()
    x["rvol20"]=v/v.rolling(20).mean()
    x["rsi14"]=_rsi(c)
    x["dist_ma20"]=c/x["ma20"]-1
    x["dist_ma60"]=c/x["ma60"]-1
    return x


def score_row(r) -> tuple[float,dict]:
    # M1 deterministic proxy score. Every feature uses information available
    # at that session close only; no future return enters the score.
    momentum=_clip(50 + 350*r["ret20"])
    trend=_clip(50 + 500*r["dist_ma60"])
    short_reversal=_clip(50 - 500*r["ret5"])
    volume=_clip(40 + 30*min(max(r["rvol20"],0),2.0))
    rsi_edge=_clip(100-abs(r["rsi14"]-55)*2.2)
    volatility=_clip(100-1200*r["vol20"])
    parts={
      "momentum":momentum,"trend":trend,"short_reversal":short_reversal,
      "volume":volume,"rsi_edge":rsi_edge,"volatility_quality":volatility,
    }
    weights={"momentum":.25,"trend":.20,"short_reversal":.15,"volume":.15,
             "rsi_edge":.10,"volatility_quality":.15}
    score=sum(parts[k]*weights[k] for k in weights)
    return round(score,2),{k:round(v,3) for k,v in parts.items()}


def backtest_ticker(ticker: str,start: str,end: str,threshold=70.0,capital_krw=10_000_000):
    df=build_features(load_history(ticker,start,end)).dropna()
    start_ts,end_ts=pd.Timestamp(start),pd.Timestamp(end)
    rows=[]
    for i in range(len(df)-1):
        date=df.index[i].tz_localize(None) if getattr(df.index[i],"tzinfo",None) else df.index[i]
        if date < start_ts or date > end_ts: continue
        score,features=score_row(df.iloc[i])
        if score < threshold: continue
        entry=float(df.iloc[i]["Close"]); exit_=float(df.iloc[i+1]["Close"])
        ret=exit_/entry-1
        rows.append(FastSignal(ticker,date.strftime("%Y-%m-%d"),score,"BUY",entry,exit_,ret,
                               round(capital_krw*ret),features))
    return rows
