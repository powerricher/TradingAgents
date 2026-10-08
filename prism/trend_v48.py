"""PRISM v4.8: frozen-signal long-trend exit and daily marked-to-market portfolio.
Research only; historical universe and strategy tuning create selection bias.
"""
import argparse, json, html
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
from .trend_v47 import atr

MODES={"E60":(60,3),"E90":(90,3),"E120":(120,3),"E90_WIDE":(90,4)}
COST=0.001
STOP_SLIPPAGE=0.002
START_CASH=10_000_000.

def load_prices(ticker,first):
    start=(pd.Timestamp(first)-pd.Timedelta(days=110)).strftime("%Y-%m-%d")
    d=yf.download(ticker,start=start,end="2026-11-01",auto_adjust=True,progress=False)
    if isinstance(d.columns,pd.MultiIndex):
        d.columns=d.columns.get_level_values(0)
    d.index=pd.DatetimeIndex(d.index).tz_localize(None).normalize()
    if d.empty or not {"Open","High","Low","Close"}.issubset(d.columns):
        raise ValueError("No usable OHLC")
    return d.sort_index()

def exit_trade(d,signal_day,mode):
    max_days,k=MODES[mode]
    ix=d.index.get_loc(signal_day)
    e=ix+1
    if e>=len(d):return None
    entry=float(d.Open.iloc[e])
    if not np.isfinite(entry) or entry<=0:return None
    av=atr(d);stop=-np.inf;peak=entry
    for j in range(e,min(len(d),e+max_days+1)):
        op,lo,cl=map(float,(d.Open.iloc[j],d.Low.iloc[j],d.Close.iloc[j]))
        # Only yesterday's established stop can execute today.
        if j>e and np.isfinite(stop):
            if op<=stop:return e,j,entry,op,"gap_stop"
            if lo<=stop:return e,j,entry,stop*(1-STOP_SLIPPAGE),"stop_touch_stress"
        if j-e>=max_days:
            return e,j,entry,op,"max_hold_next_open"
        peak=max(peak,cl)
        ma20=float(d.Close.iloc[max(0,j-19):j+1].mean())
        if cl<ma20 and j+1<len(d):
            return e,j+1,entry,float(d.Open.iloc[j+1]),"ma20_next_open"
        if pd.notna(av.iloc[j]):
            stop=max(stop,peak-k*float(av.iloc[j]))
    return None

def portfolio(signals,ohlc,mode,rotation=False):
    # Daily chronological event engine; no same-session reentry after a sale.
    candidates={}
    for s in signals:
        candidates.setdefault(s["entry_date"],[]).append(s)
    for day in candidates:
        candidates[day].sort(key=lambda x:(-x["edge"],x["ticker"]))
    days=sorted(set().union(*(set(d.index) for d in ohlc.values())))
    cash=START_CASH;position=None;curve=[];ledger=[];skipped=0
    for day in days:
        key=day.strftime("%Y-%m-%d");sold=False
        if position is not None and key==position["exit_date"]:
            cash=position["shares"]*position["exit_price"]*(1-COST/2)
            ledger.append(dict(mode=mode,rotation=rotation,ticker=position["ticker"],entry_date=position["entry_date"],exit_date=key,capital_after=cash,reason=position["reason"]))
            position=None;sold=True
        available=candidates.get(key,[])
        # A rotation signal was generated at prior close and enters today open.
        # Require materially higher frozen Edge than the currently held position.
        if rotation and position and available:
            better=available[0]
            if better["ticker"]!=position["ticker"] and better["edge"]>max(0,position["edge"])*1.5:
                held=ohlc[position["ticker"]]
                if day in held.index:
                    cash=position["shares"]*float(held.loc[day,"Open"])*(1-COST/2)
                    ledger.append(dict(mode=mode,rotation=True,ticker=position["ticker"],entry_date=position["entry_date"],exit_date=key,capital_after=cash,reason="rotation_open"))
                    position=None
                    # This is an open-to-open switch, unlike a scheduled exit.
        if position is None and available and not sold:
            best=available[0]
            if best["ticker"] in ohlc:
                shares=cash*(1-COST/2)/best["entry_price"]
                position={**best,"shares":shares}
                cash=0.
        skipped+=len(available)-(1 if position and position["entry_date"]==key else 0)
        equity=cash
        if position:
            d=ohlc[position["ticker"]]
            if day in d.index:
                equity=position["shares"]*float(d.loc[day,"Close"])
            else:
                # Missing valuation: never silently use zero or forward-fill.
                raise ValueError("Missing close for "+position["ticker"]+" "+key)
        curve.append(dict(date=key,mode=mode,rotation=rotation,equity=equity))
    if position is not None:
        raise ValueError("Unclosed position at evaluation horizon: "+position["ticker"])
    vals=np.array([r["equity"] for r in curve])
    peak=np.maximum.accumulate(np.r_[START_CASH,vals])[1:]
    mdd=float(np.min(vals/peak-1))
    return {"final_krw":float(cash),"return":float(cash/START_CASH-1),"daily_close_mdd":mdd,"trades":len(ledger),"skipped":int(skipped)},ledger,curve

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--signals",default="results/prism-v45/report.json")
    ap.add_argument("--out",default="results/prism-v48")
    args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    selected=pd.DataFrame(json.loads(Path(args.signals).read_text())["selected_rows"])
    ohlc={};rows=[];issues=[]
    for ticker,g in selected.groupby("ticker"):
        try:
            d=load_prices(ticker,g.date.min());ohlc[ticker]=d
            for s in g.itertuples(index=False):
                dt=pd.Timestamp(s.date)
                if dt not in d.index:
                    issues.append([ticker,str(dt.date()),"missing_signal"]);continue
                for mode in MODES:
                    z=exit_trade(d,dt,mode)
                    if z is None:
                        issues.append([ticker,str(dt.date()),mode+" incomplete"]);continue
                    e,x,entry,exit_price,reason=z
                    rows.append(dict(ticker=ticker,mode=mode,entry_date=str(d.index[e].date()),exit_date=str(d.index[x].date()),entry_price=entry,exit_price=exit_price,edge=float(s.edge),reason=reason,ret=exit_price/entry-1))
        except Exception as ex:
            issues.append([ticker,"download",str(ex)])
    trades=pd.DataFrame(rows)
    if trades.empty:raise RuntimeError("No completed trades")
    report={"version":"4.8","status":"EXPERIMENTAL","signal_count":len(selected),"completed_by_mode":trades.groupby("mode").size().to_dict(),"issues":issues,"strategies":{},"limitations":["Frozen D+1 entries, not a D+5 trained model","Historical fixed 50-stock universe is not point-in-time","Daily close MDD excludes intraday lows","Stop touch haircut 20bps is a scenario, not real execution","Rotation threshold selected for research, not independent OOS","Missing trades or open positions invalidate affected strategy"]}
    ledgers=[];curves=[]
    for mode in MODES:
        sub=trades[trades["mode"]==mode].to_dict("records")
        if len(sub)!=len(selected):
            report["strategies"][mode]={"status":"INCOMPLETE","completed":len(sub)}
            continue
        for rotation in (False,True):
            key=mode+("_ROT" if rotation else "")
            try:
                result,ledger,curve=portfolio(sub,ohlc,mode,rotation)
                report["strategies"][key]=result
                for x in ledger:x["strategy"]=key
                for x in curve:x["strategy"]=key
                ledgers.extend(ledger);curves.extend(curve)
            except Exception as ex:
                report["strategies"][key]={"status":"INVALID","error":str(ex)}
    trades.to_csv(out/"trades.csv",index=False)
    pd.DataFrame(ledgers).to_csv(out/"ledger.csv",index=False)
    pd.DataFrame(curves).to_csv(out/"daily_equity.csv",index=False)
    (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    lines="".join("<tr><td>"+html.escape(k)+"</td><td>"+(f"{v['return']:.2%}" if "return" in v else "INVALID")+"</td><td>"+(f"{v['daily_close_mdd']:.2%}" if "daily_close_mdd" in v else "—")+"</td><td>"+str(v.get("trades","—"))+"</td></tr>" for k,v in report["strategies"].items())
    (out/"report.html").write_text("<!doctype html><meta charset='utf-8'><title>PRISM v4.8</title><style>body{background:#091424;color:#e6f2ff;font:15px system-ui;margin:30px}h1{color:#6ee7b7}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #345}</style><h1>PRISM v4.8 · Extended Trend</h1><p>Research challenger · 10M KRW initial capital · round trip 10bps · stop touch stress 20bps</p><table><tr><th>Strategy</th><th>Return</th><th>Daily close MDD</th><th>Trades</th></tr>"+lines+"</table><p>Check summary.json for issues and limitations.</p>",encoding="utf-8")
    print(json.dumps({"strategies":report["strategies"],"issues":len(issues)},ensure_ascii=False))
if __name__=="__main__":main()
