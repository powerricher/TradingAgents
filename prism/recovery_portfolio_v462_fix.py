"""PRISM v4.62 corrected one-slot event engine.
Supports entry-day stop exits without altering the shared legacy portfolio.
"""
import numpy as np,pandas as pd
from .trend_v48 import COST,START_CASH

def portfolio_same_day(signals,ohlc,mode):
    candidates={}
    for s in signals:
        candidates.setdefault(str(s["entry_date"]),[]).append(s)
    for key in candidates:
        candidates[key].sort(key=lambda x:(-float(x["edge"]),str(x["ticker"])))
    days=sorted(set().union(*(set(d.index) for d in ohlc.values())))
    cash=float(START_CASH);position=None;curve=[];ledger=[];skipped=0
    for day in days:
        key=day.strftime("%Y-%m-%d")
        sold=False
        # Existing position may exit at this session's open or intraday.
        if position is not None and position["exit_date"]==key:
            cash=position["shares"]*float(position["exit_price"])*(1-COST/2)
            ledger.append({"mode":mode,"rotation":False,"ticker":position["ticker"],
                "entry_date":position["entry_date"],"exit_date":key,
                "capital_after":cash,"reason":position["reason"]})
            position=None;sold=True
        available=candidates.get(key,[])
        if position is None and not sold and available:
            best=available[0]
            if best["ticker"] not in ohlc:raise RuntimeError("Missing ticker "+best["ticker"])
            price=float(best["entry_price"])
            if not np.isfinite(price) or price<=0:raise RuntimeError("Invalid entry price")
            shares=cash*(1-COST/2)/price
            position={**best,"shares":shares}
            cash=0.
            # After entry at the open, the fixed stop can be hit intraday.
            # This is a single daily-OHLC fill assumption, not tick-level evidence.
            if position["exit_date"]==key:
                cash=shares*float(position["exit_price"])*(1-COST/2)
                ledger.append({"mode":mode,"rotation":False,"ticker":position["ticker"],
                    "entry_date":key,"exit_date":key,"capital_after":cash,
                    "reason":position["reason"]})
                position=None;sold=True
        skipped+=len(available)-(1 if (position is not None and position["entry_date"]==key) or
            (sold and ledger and ledger[-1]["entry_date"]==key) else 0)
        equity=cash
        if position is not None:
            d=ohlc[position["ticker"]]
            if day not in d.index:raise RuntimeError("Missing close "+position["ticker"]+" "+key)
            equity=position["shares"]*float(d.loc[day,"Close"])
        curve.append({"date":key,"mode":mode,"rotation":False,"equity":float(equity)})
    if position is not None:raise RuntimeError("Unclosed position at horizon: "+position["ticker"])
    vals=np.asarray([x["equity"] for x in curve])
    peak=np.maximum.accumulate(np.r_[START_CASH,vals])[1:]
    return ({"final_krw":float(cash),"return":float(cash/START_CASH-1),
        "daily_close_mdd":float(np.min(vals/peak-1)),
        "trades":len(ledger),"skipped":int(skipped),
        "same_day_exits":sum(x["entry_date"]==x["exit_date"] for x in ledger)},
        ledger,curve)
