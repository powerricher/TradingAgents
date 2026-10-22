"""PRISM v4.82: D4 inverse ETF regime strategy research.
Daily OHLC next-open execution proxy, not independent OOS or investment advice.
"""
import json, html
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

START=10_000_000.
ETFS={"PSQ":"QQQ","SQQQ":"QQQ","SOXS":"SOXX"}
COST=.001
SLIP=.002
END="2026-10-07"
def prices(ticker):
    d=yf.download(ticker,start="2021-01-01",end=END,auto_adjust=True,progress=False,threads=False)
    if isinstance(d.columns,pd.MultiIndex):d.columns=d.columns.get_level_values(0)
    if d.empty:raise RuntimeError("No prices: "+ticker)
    d=d.dropna(subset=["Open","High","Low","Close"]).copy()
    d.index=pd.DatetimeIndex(d.index).tz_localize(None).normalize()
    return d
def run(etf,benchmark,d,base,hold,stop):
    # Signals formed at prior close; no use of today's closing price for today's open.
    b=base.Close.astype(float)
    ma50=b.rolling(50,min_periods=50).mean()
    ma200=b.rolling(200,min_periods=200).mean()
    mom20=b.pct_change(20)
    bearish=(b<ma200)&(b<ma50)&(mom20<0)
    signals=bearish.shift(1).fillna(False).astype(bool)
    days=d.index.intersection(base.index)
    days=days[(days>=pd.Timestamp("2022-01-01"))&(days<=pd.Timestamp("2026-10-06"))]
    if len(days)<200:raise RuntimeError("Too few trading days "+etf)
    cash=START;shares=0.;entry=0.;entry_i=-1;entry_day=None;ledger=[];curve=[]
    for i,day in enumerate(days):
        op=float(d.loc[day,"Open"]);lo=float(d.loc[day,"Low"]);cl=float(d.loc[day,"Close"])
        if shares:
            stop_price=entry*(1-stop)
            exit_price=None;reason=None
            if i>entry_i and op<=stop_price:exit_price=op;reason="gap_stop"
            elif lo<=stop_price:exit_price=stop_price*(1-SLIP);reason="stop_touch"
            elif i-entry_i>=hold:exit_price=cl;reason="time_close"
            if exit_price is not None:
                cash=shares*exit_price*(1-COST/2)
                ledger.append({"etf":etf,"benchmark":benchmark,"entry_date":str(entry_day.date()),"exit_date":str(day.date()),"entry_price":entry,"exit_price":exit_price,"reason":reason,"capital_after":cash})
                shares=0.
        # no same-day re-entry after any liquidation
        sold=bool(ledger and ledger[-1]["exit_date"]==str(day.date()))
        if not shares and not sold and bool(signals.get(day,False)):
            entry=op;entry_i=i;entry_day=day
            shares=cash*(1-COST/2)/entry;cash=0.
            stop_price=entry*(1-stop)
            if lo<=stop_price:
                cash=shares*stop_price*(1-SLIP)*(1-COST/2)
                ledger.append({"etf":etf,"benchmark":benchmark,"entry_date":str(day.date()),"exit_date":str(day.date()),"entry_price":entry,"exit_price":stop_price*(1-SLIP),"reason":"entry_day_stop","capital_after":cash})
                shares=0.
        equity=cash+shares*cl
        curve.append({"date":str(day.date()),"equity":float(equity)})
    # Mark-to-market terminal open position rather than fabricate a sale.
    terminal_open=bool(shares)
    vals=np.array([x["equity"] for x in curve])
    peak=np.maximum.accumulate(np.r_[START,vals])[1:]
    result={"return":float(vals[-1]/START-1),"mdd":float(np.min(vals/peak-1)),
            "final_krw":float(vals[-1]),"trades":len(ledger),"terminal_open_position":terminal_open}
    return result,ledger,curve

def main():
    out=Path("results/prism-v482");out.mkdir(parents=True,exist_ok=True)
    data={t:prices(t) for t in sorted(set(ETFS)|set(ETFS.values()))}
    results={};ledgers=[];curves=[];yearly=[]
    for etf,benchmark in ETFS.items():
      for hold in (1,3,5):
       for stop in (.04,.07):
        name=f"{etf}_H{hold}_SL{int(stop*100)}"
        perf,ledger,curve=run(etf,benchmark,data[etf],data[benchmark],hold,stop)
        results[name]=perf
        for x in ledger:x["strategy"]=name
        for x in curve:x["strategy"]=name
        ledgers.extend(ledger);curves.extend(curve)
        eq=pd.DataFrame(curve)
        eq["date"]=pd.to_datetime(eq.date)
        for year,g in eq.groupby(eq.date.dt.year):
            before=eq[eq.date<g.date.min()]
            initial=float(before.equity.iloc[-1]) if not before.empty else START
            yearly.append({"strategy":name,"year":int(year),"return":float(g.equity.iloc[-1]/initial-1)})
    pd.DataFrame(ledgers).to_csv(out/"ledger.csv",index=False)
    pd.DataFrame(curves).to_csv(out/"daily_equity.csv",index=False)
    pd.DataFrame(yearly).to_csv(out/"yearly_returns.csv",index=False)
    report={"version":"v4.82","status":"D4_INVERSE_ETF_HISTORICAL_NOT_OOS","strategies":results,
      "rule":"Benchmark prior-close below MA50 and MA200 with negative 20-day momentum; enter inverse ETF next open",
      "limitations":["Leveraged ETFs reset daily and can decay; multi-day holding returns differ from inverse index multiples.",
       "Historical rule selection is in-sample, not independent OOS.",
       "One full-capital position; stop-first daily OHLC assumptions; no actual intraday execution.",
       "Transaction cost 10bps round trip, additional 20bps haircut on stop touches.",
       "Terminal open position is marked to market, not assumed liquidated.",
       "No tax, financing, spread, market impact or broker order constraints.",
       "No shared ABCD portfolio yet. Annual 200% is a research target, not a forecast."]}
    (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    trs="".join("<tr><td>%s</td><td>%.2f%%</td><td>%.2f%%</td><td>%d</td></tr>"%(html.escape(k),100*v["return"],100*v["mdd"],v["trades"]) for k,v in results.items())
    (out/"report.html").write_text("<!doctype html><html lang='ko'><meta charset='utf-8'><title>PRISM v4.82</title><style>body{background:#0b1729;color:#eaf4ff;font:15px system-ui;margin:30px}h1{color:#5de4bd}td,th{padding:10px;border-bottom:1px solid #345}</style><h1>PRISM v4.82 D4 Inverse ETF</h1><p>Historical, not OOS or live trading.</p><table><tr><th>Strategy</th><th>Return</th><th>MDD</th><th>Trades</th></tr>"+trs+"</table></html>",encoding="utf-8")
    print(json.dumps({"status":"V482_COMPLETE","strategies":len(results)}))
if __name__=="__main__":main()
