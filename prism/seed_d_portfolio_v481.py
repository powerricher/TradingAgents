"""PRISM v4.81 Seed D3 BEAR rebound portfolio replay.
Historical research; daily OHLC stop-first approximation; NOT OOS.
"""
import json,html
from pathlib import Path
import numpy as np,pandas as pd,yfinance as yf
from .regime_adaptive_v436 import regimes
from .seed_c_multi_position_v471 import simulate
from .trend_v48 import COST
START=10_000_000.
def load_batch(names):
    prices={};errors=[]
    for j in range(0,len(names),25):
        batch=names[j:j+25]
        try:
            z=yf.download(batch,start="2021-01-01",end="2026-10-07",auto_adjust=True,progress=False,threads=False,group_by="ticker")
            for t in batch:
                try:
                    d=(z[t] if isinstance(z.columns,pd.MultiIndex) else z).dropna(subset=["Open","High","Low","Close"]).copy()
                    d.index=pd.DatetimeIndex(d.index).tz_localize(None).normalize()
                    if d.empty:raise ValueError("Empty prices")
                    prices[t]=d
                except Exception as e:errors.append({"ticker":t,"error":str(e)})
        except Exception as e:errors.extend({"ticker":t,"error":str(e)} for t in batch)
    return prices,errors
def exit_path(d,entry,price,sl,hold):
    ix=d.index.get_loc(entry);stop=price*(1-sl)
    for j in range(ix,min(ix+hold,len(d)-1)+1):
        day=d.index[j];op=float(d.Open.iloc[j]);lo=float(d.Low.iloc[j])
        if j>ix and op<=stop:return day,op,"gap_stop"
        if lo<=stop:return day,stop*.998,"stop_touch_stress"
        if j-ix==hold:return day,float(d.Close.iloc[j]),"time_close"
    return None
def main():
    out=Path("results/prism-v481");out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv("results/prism-v479/signal_forward_returns.csv")
    df=df[df.detector=="D3_OVERSOLD_LONG_PROXY"].copy()
    if df.empty:raise RuntimeError("Missing D3 candidates")
    df["entry_date"]=pd.to_datetime(df.entry_date).dt.normalize()
    market=regimes(pd.Timestamp("2022-01-01"))
    df["regime"]=df.entry_date.map(market)
    if df.regime.isna().any() or (df.regime=="UNKNOWN").any():raise RuntimeError("Unknown regime")
    df=df[df.regime=="BEAR"].copy()
    if df.empty:raise RuntimeError("No BEAR D3 signals")
    prices,errors=load_batch(sorted(df.ticker.unique()))
    if errors:raise RuntimeError("Incomplete price coverage "+json.dumps(errors[:10]))
    calendar=list(pd.bdate_range("2022-01-03","2026-10-06"))
    # Use QQQ's actual trading sessions instead of business-day holidays.
    q=yf.download("QQQ",start="2022-01-01",end="2026-10-07",auto_adjust=True,progress=False)
    if isinstance(q.columns,pd.MultiIndex):q.columns=q.columns.get_level_values(0)
    calendar=list(pd.DatetimeIndex(q.dropna(subset=["Close"]).index).tz_localize(None).normalize())
    report={"version":"v4.81","status":"HISTORICAL_SEED_D3_PORTFOLIO_NOT_OOS",
      "signals":len(df),"tickers":df.ticker.nunique(),"strategies":{},
      "limitations":["QQQ BEAR based on prior-close regime, no future regime labels.",
       "Next-open entries, daily OHLC stop-first; no intraday sequencing.",
       "Close-time exits assume fill at observed close; not guaranteed.",
       "Current-listed 2500-name universe, delistings absent, survivorship bias.",
       "D3 signal selected after inspecting historical outcomes; not independent OOS.",
       "No short positions, D4 ETF or shared-cash ABCD replay.",
       "Existing simulator applies 10bps round-trip costs and 20bps stop-touch haircut.",
       "Unexecuted signals and capital occupancy are accounted for by the simulator."]}
    ledgers=[];curves=[];yearly=[]
    for hold in (3,5):
      for sl in (.03,.05,.07):
        records=[]
        for r in df.itertuples(index=False):
            d=prices[r.ticker];entry=r.entry_date
            if entry not in d.index:raise RuntimeError("Missing entry "+r.ticker)
            price=float(d.loc[entry,"Open"])
            if abs(price/float(r.entry_open)-1)>.001:raise RuntimeError("Price mismatch "+r.ticker)
            ex=exit_path(d,entry,price,sl,hold)
            if ex is None:raise RuntimeError("Unresolved exit "+r.ticker)
            day,px,reason=ex
            records.append({"ticker":r.ticker,"entry_date":str(entry.date()),"exit_date":str(day.date()),
             "entry_price":price,"exit_price":px,"edge":float(r.signal_rvol),"reason":reason})
        name=f"D3_BEAR_H{hold}_SL{int(sl*100)}"
        perf,ledger,curve=simulate(records,calendar,prices,4,.25)
        report["strategies"][name]=perf
        for x in ledger:x["strategy"]=name
        for x in curve:x["strategy"]=name
        ledgers+=ledger;curves+=curve
        eq=pd.DataFrame(curve);eq["date"]=pd.to_datetime(eq.date)
        for year,g in eq.groupby(eq.date.dt.year):
            before=eq[eq.date<g.date.min()]
            initial=float(before.equity.iloc[-1]) if not before.empty else START
            yearly.append({"strategy":name,"year":int(year),"return":float(g.equity.iloc[-1]/initial-1)})
    pd.DataFrame(ledgers).to_csv(out/"ledger.csv",index=False)
    pd.DataFrame(curves).to_csv(out/"daily_equity.csv",index=False)
    pd.DataFrame(yearly).to_csv(out/"yearly_returns.csv",index=False)
    (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    trs="".join("<tr><td>%s</td><td>%.2f%%</td><td>%.2f%%</td><td>%s</td></tr>"%(html.escape(k),100*v["return"],100*v["mdd"],v["trades"]) for k,v in report["strategies"].items())
    (out/"report.html").write_text("<!doctype html><html lang='ko'><meta charset='utf-8'><title>PRISM v4.81</title><style>body{background:#0b1729;color:#eaf4ff;font:15px system-ui;margin:30px}h1{color:#5de4bd}td,th{padding:10px;border-bottom:1px solid #345}</style><h1>PRISM v4.81 Seed D3 Portfolio</h1><p>Historical research, not independent OOS or shared-cash ABCD.</p><table><tr><th>Policy</th><th>Return</th><th>MDD</th><th>Trades</th></tr>"+trs+"</table></html>",encoding="utf-8")
    print(json.dumps({"status":"V481_COMPLETE","strategies":report["strategies"]}))
if __name__=="__main__":main()
